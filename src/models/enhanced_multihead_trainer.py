"""
Enhanced Multihead Trainer với CA và Domain Adaptation
Modifications cho multihead_trainer.py
"""
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
import warnings
from tqdm import tqdm
from collections import defaultdict

try:
    from enhanced_embedding_model import create_enhanced_model
    from enhanced_multihead_model import EnhancedMultiheadModel
    from enhanced_losses import EnhancedMultiheadLoss
    from grl_domain_classifier import compute_lambda
    from metric_utils import HardNegativePairSelector, RandomNegativeTripletSelector
    from metrics import MetricsCollection, classification_accuracy
except ImportError:
    pass


def init_enhanced_model(args, label_encoder, use_coord_attention=True, use_domain_adaptation=True):
    """
    Khởi tạo Enhanced Model với CA và GRL
    
    Args:
        args: arguments
        label_encoder: label encoder
        use_coord_attention: enable Coordinate Attention
        use_domain_adaptation: enable Domain Adaptation
    
    Returns:
        model, device
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_classes = len(label_encoder.classes_)
    print(f"n_classes={n_classes}")
    
    # Create Enhanced Embedding Model
    E_model = create_enhanced_model(
        args,
        use_coord_attention=use_coord_attention,
        use_domain_adaptation=use_domain_adaptation
    )
    
    # Wrap với Multihead Model
    model = EnhancedMultiheadModel(
        E_model, 
        n_classes, 
        train_with_side_labels=args.train_with_side_labels,
        return_domain_logits=use_domain_adaptation
    )
    
    print(model)
    
    if args.load_mod:
        model.load_state_dict(torch.load(args.load_mod))
    
    model.to(device)
    
    return model, device


def train_enhanced_model(model, optimizer, scheduler,
                        device, dataloaders,
                        results_dir,
                        label_encoder,
                        criterion,
                        num_epochs=100,
                        earlystop_patience=7,
                        simul_sidepairs=False,
                        train_with_side_labels=True,
                        sidepairs_agg='post_mean',
                        metric_evaluator_type='euclidean',
                        val_evaluator='metric',
                        use_domain_adaptation=True):
    """
    Training loop với domain adaptation
    Modified từ hneg_train_model trong multihead_trainer.py
    """
    from azureml.core.run import Run
    run = Run.get_context()
    
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    has_waited = 0
    stop_training = False
    
    epoch_metrics = MetricsCollection()
    
    # Import evaluators
    from metric_test_eval import MetricEmbeddingEvaluator, LogitEvaluator
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        # Update lambda cho GRL (tăng dần theo epoch)
        if use_domain_adaptation:
            lambda_ = compute_lambda(epoch, num_epochs)
            model.set_domain_lambda(lambda_)
            print(f"Domain lambda: {lambda_:.4f}")
        
        # Evaluators
        evaluator = MetricEmbeddingEvaluator(
            model, 
            simul_sidepairs=simul_sidepairs, 
            sidepairs_agg_method=sidepairs_agg, 
            metric_evaluator_type=metric_evaluator_type
        )
        logit_evaluator = LogitEvaluator(
            model, 
            simul_sidepairs=simul_sidepairs, 
            sidepairs_agg_method=sidepairs_agg
        )
        
        # Training and validation
        for phase in ['train', 'val']:
            print('Phase: {}'.format(phase))
            
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            batch_metrics = MetricsCollection()
            distance_records = defaultdict(list)
            
            loader = dataloaders[phase]
            pbar = tqdm(loader, total=len(loader), 
                       desc=f"Epoch {epoch} {phase}", ncols=0, disable=None)
            
            for batch_index, batch_data in enumerate(pbar):
                inputs = batch_data['image'].to(device)
                labels = batch_data['label'].to(device)
                is_front = batch_data.get('is_front', None)
                is_ref = batch_data.get('is_ref', None)
                
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    all_outputs = model(inputs, labels)
                    
                    # Compute losses (bao gồm domain loss)
                    loss_outputs = criterion(
                        all_outputs, 
                        labels, 
                        is_front=is_front, 
                        is_ref=is_ref,
                        domain_logits=all_outputs.get('domain_logits', None)
                    )
                    
                    if loss_outputs is None:
                        warnings.warn("loss_outputs is None, skip batch")
                        continue
                    
                    # Classification accuracy
                    logits = all_outputs['logits']
                    if train_with_side_labels:
                        logits = model.shift_label_indexes(logits)
                    
                    accuracies = classification_accuracy(logits, labels, topk=(1, 5))
                    batch_metrics.add(phase, 'acc1', accuracies[0].item(), inputs.size(0))
                    batch_metrics.add(phase, 'acc5', accuracies[1].item(), inputs.size(0))
                    
                    # Record distances cho metrics
                    for prefix in ['triplet_', 'contrastive_']:
                        for n in ['distances', 'targets']:
                            k = prefix + n
                            if k in loss_outputs:
                                distance_records[k].append(loss_outputs[k])
                    
                    # Backward pass
                    if phase == 'train':
                        loss_outputs['loss'].backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        
                        lr = optimizer.param_groups[0]['lr']
                        batch_metrics.add(phase, 'lr', lr, inputs.size(0))
                
                # Log losses
                loss_keys = ['loss', 'metric_loss', 'ce', 'arcface', 
                            'contrastive', 'triplet', 'focal', 'domain']
                for k in loss_keys:
                    if k in loss_outputs:
                        batch_metrics.add(phase, k, loss_outputs[k].item(), inputs.size(0))
                
                # Log domain accuracy
                if 'domain_acc' in loss_outputs:
                    batch_metrics.add(phase, 'domain_acc', 
                                    loss_outputs['domain_acc'].item(), inputs.size(0))
                
                # Update progress bar
                pbar_dict = {}
                for k, meter in batch_metrics[phase].items():
                    fmt = "{:.1e}" if k == 'lr' else "{:.2f}"
                    pbar_dict[k.replace("contrastive", "c").replace("triplet", "t")] = fmt.format(meter.avg)
                pbar.set_postfix(**pbar_dict)
            
            # End of epoch phase
            for key, meter in batch_metrics[phase].items():
                epoch_metrics.add(phase, key, meter.avg, 1)
                run.log(f'{phase}_{key}', meter.avg)
            
            # Evaluation checkpoint
            checkpoint = 5
            if phase == 'val' and epoch % checkpoint == 0:
                print("#### Checkpoint ###")
                
                # Evaluate
                if 'logit' in val_evaluator:
                    print("Evaluating logit metrics")
                    logit_evaluator.multihead_model = model
                    metrics_results, _ = logit_evaluator.eval_model(device, dataloaders)
                    
                    for key, value in metrics_results.items():
                        if isinstance(value, (int, float)):
                            epoch_metrics.add(phase, key, value, 1)
                            run.log(f'{phase}_{key}_logit', value)
                
                if 'metric' in val_evaluator:
                    print("Evaluating metric metrics")
                    evaluator.siamese_model = model.embedding_model
                    metrics_results, _ = evaluator.eval_model(device, dataloaders)
                    
                    for key, value in metrics_results.items():
                        if isinstance(value, (int, float)):
                            epoch_metrics.add(phase, key, value, 1)
                            run.log(f'{phase}_{key}_metric', value)
                
                # Check for best model
                best_value, best_idx = epoch_metrics['val']['micro-ap'].best(mode='max')
                if best_idx + 1 == len(epoch_metrics['val']['micro-ap'].history):
                    has_waited = 1
                    best_model_wts = copy.deepcopy(model.state_dict())
                    print(f"Saving best model: {best_value:.4f}")
                else:
                    has_waited += 1
                    if has_waited >= earlystop_patience:
                        print(f"Early stop after {has_waited} waits")
                        stop_training = True
                
                # Scheduler step
                if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(-1.0 * epoch_metrics['val']['micro-ap'].value)
                else:
                    scheduler.step()
        
        print()
        if stop_training:
            break
    
    # Training complete
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    
    # Get best metrics
    _, best_epoch = epoch_metrics['val']['micro-ap'].best(mode='max')
    best_metrics = {'best_epoch': best_epoch}
    for k, v in epoch_metrics['val'].items():
        try:
            best_metrics[k] = v.history[best_epoch]
        except:
            pass
    
    # Load best weights
    model.load_state_dict(best_model_wts)
    
    return model, best_metrics


if __name__ == '__main__':
    print("Enhanced trainer module loaded successfully!")
    print("To use:")
    print("1. Import: from enhanced_multihead_trainer import init_enhanced_model, train_enhanced_model")
    print("2. Replace init_mod_dev with init_enhanced_model")
    print("3. Replace hneg_train_model with train_enhanced_model")