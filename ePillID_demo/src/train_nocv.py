import os
import torch
import numpy as np
import pandas as pd
import torchvision
from torchvision import datasets, models as torch_models, transforms
import datetime
import time
import sys
import copy
import warnings
import pickle
import logging
from sklearn.preprocessing import LabelEncoder
import torch.optim as optim
from torch.optim import lr_scheduler

# Import các module đánh giá
from metric_test_eval import MetricEmbeddingEvaluator, LogitEvaluator

# ========== ENHANCED IMPORTS ==========
# Lưu ý: Đảm bảo các file này tồn tại hoặc sửa lại đường dẫn nếu cần
try:
    from models.enhanced_embedding_model import create_enhanced_model
    from models.enhanced_multihead_model import EnhancedMultiheadModel
    from models.enhanced_losses import EnhancedMultiheadLoss
    from models.grl_domain_classifier import compute_lambda
except ImportError:
    # Fallback nếu chưa có file enhanced, dùng file gốc
    from models.multihead_models import MultiheadModel as EnhancedMultiheadModel
    from losses import MultiheadLoss as EnhancedMultiheadLoss

from metric_utils import HardNegativePairSelector, RandomNegativeTripletSelector
# ======================================

logger = logging.getLogger(__name__)

def run(args):
    if args.supress_warnings:
        warnings.simplefilter("ignore")

    def adjust_path(p):
        return os.path.join(args.data_root_dir, p)

    # Điều chỉnh đường dẫn
    args.label_encoder = adjust_path(args.label_encoder)
    args.all_imgs_csv = adjust_path(args.all_imgs_csv)
    args.val_imgs_csv = adjust_path(args.val_imgs_csv)
    args.test_imgs_csv = adjust_path(args.test_imgs_csv)
    args.results_dir = adjust_path(args.results_dir)

    print(args)

    # ========== IMPORT TRAINER ==========
    # Sử dụng hàm hneg_train_model từ file multihead_trainer đã sửa lỗi
    from multihead_trainer import torch_transform, create_dataloaders, hneg_train_model
    # Nếu bạn muốn dùng hàm init riêng, hãy import ở đây, nếu không dùng init mặc định
    from multihead_trainer import init_mod_dev 
    # ====================================

    def build_logid_string(args, add_timestamp=True):
        param_str = "lr{}_dr{}_lrpatience{}_lrfactor{}_{}".format(
            args.init_lr, args.dropout, args.lr_patience,
            args.lr_factor, args.appearance_network)

        if add_timestamp:
            param_str += "_" + datetime.datetime.now().strftime("%Y%m%d%H%M")

        return param_str

    param_str = build_logid_string(args)

    # Azure ML
    try:
        from azureml.core.run import Run
        run = Run.get_context()
    except ImportError:
        class MockRun:
            def log(self, k, v): print(f"Log: {k}={v}")
            def log_image(self, name, plot): pass
            def tag(self, k, v): pass
        run = MockRun()

    # Log arguments
    if not hasattr(args, 'folds_csv_dir'):
        for k, v in vars(args).items():
            run.tag(k, str(v))

    save_path = os.path.join(args.results_dir, param_str)
    os.makedirs(save_path, exist_ok=True)
    print("save_path", save_path)

    logger.info(f"cuda.is_available={torch.cuda.is_available()}, n_gpu={torch.cuda.device_count()}")

    # --- LABEL ENCODER ---
    if not os.path.exists(args.label_encoder):
        logger.warning(f"Fitting a new label encoder at {args.label_encoder}")
        all_imgs_df = pd.read_csv(args.all_imgs_csv)
        label_encoder = LabelEncoder()
        label_encoder.fit(all_imgs_df['label'])
        pickle.dump(label_encoder, open(args.label_encoder, "wb"))
    else:
        logger.info(f"Loading label encoder: {args.label_encoder}")
        with open(args.label_encoder, 'rb') as pickle_file:
            label_encoder = pickle.load(pickle_file)

    logger.info(f"label_encoder.classes_={label_encoder.classes_}")    
    logger.info("The label encoder has {} classes.".format(len(label_encoder.classes_)))

    # --- DATA LOADING ---
    all_images_df = pd.read_csv(args.all_imgs_csv)
    val_df = pd.read_csv(args.val_imgs_csv)
    test_df = pd.read_csv(args.test_imgs_csv)

    for df in [all_images_df, val_df, test_df]:
        df['image_path'] = df['image_path'].apply(lambda x: os.path.join(args.data_root_dir, args.img_dir, x))

    val_test_image_paths = list(val_df['image_path'].values) + list(test_df['image_path'].values)
    train_df = all_images_df[~all_images_df['image_path'].isin(val_test_image_paths)]

    ref_only_df = train_df[train_df['is_ref']]
    cons_train_df = train_df[train_df['is_ref'] == False]
    cons_val_df = val_df

    print("all_images", len(all_images_df), "train", len(train_df), "val", len(val_df), "test", len(test_df))
    run.log("all_images_size", len(all_images_df))
    run.log("train_size", len(train_df))
    run.log("val_size", len(val_df))
    run.log("test_size", len(test_df))

    import classif_utils
    classif_utils.ClassificationDataset.set_datadir(os.path.join(args.data_root_dir, args.img_dir))

    # --- HELPER FUNCTIONS ---
    def plot_pr_curve(plt, dataset_name):
        run.log_image(name='{}_{}_{}'.format(
                    dataset_name,
                    datetime.datetime.now().strftime("%H:%M:%S"),
                    'PR-curve'
                    ), plot=plt)
        plt.close()

    def log_metrics(metrics_results, dataset_name):
        from metrics import create_prec_inds_str
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        run_metrics = []
        for k, v in metrics_results.items():
            if ('p_indices' in k) and not ('sanity' in dataset_name):
                pind_str = create_prec_inds_str(v, label_encoder)
                run.log("{}_{}".format(dataset_name, k), pind_str)
                run_metrics.append([os.path.split(args.val_imgs_csv)[1], dataset_name, k, pind_str])
            elif isinstance(v, (int, float)):
                run.log("{}_{}".format(dataset_name, k), v)
                run_metrics.append([os.path.split(args.val_imgs_csv)[1], dataset_name, k, v])
        return run_metrics

    # ========== ENHANCED TRAINING PIPELINE ==========
    print("\n" + "="*60)
    print("ENHANCED TRAINING START")
    print(f" - Use Coordinate Attention: {getattr(args, 'use_coord_attention', True)}")
    print(f" - Use Domain Adaptation: {getattr(args, 'use_domain_adaptation', True)}")
    print("="*60 + "\n")
    
    # 1. Init Enhanced Model (sử dụng EnhancedEmbeddingModel + EnhancedMultiheadModel)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_classes = len(label_encoder.classes_)
    print(f"n_classes={n_classes}")
    
    # Import enhanced modules
    from models.enhanced_embedding_model import create_enhanced_model
    from models.enhanced_multihead_model import EnhancedMultiheadModel
    
    # Create Enhanced Embedding Model
    E_model = create_enhanced_model(
        args,
        use_coord_attention=getattr(args, 'use_coord_attention', True),
        use_domain_adaptation=getattr(args, 'use_domain_adaptation', True)
    )
    
    # Wrap với Enhanced Multihead Model
    model = EnhancedMultiheadModel(
        E_model, 
        n_classes, 
        train_with_side_labels=args.train_with_side_labels,
        return_domain_logits=getattr(args, 'use_domain_adaptation', True)
    )
    
    print(model)
    
    if args.load_mod:
        model.load_state_dict(torch.load(args.load_mod))
    
    model.to(device)
    
    # 2. Setup optimizer
    if args.optimizer == 'momentum':
        optimizer = optim.SGD(list(model.parameters()), lr=args.init_lr)
    elif args.optimizer == 'adamdelta':
        optimizer = optim.Adadelta(model.parameters(), lr=args.init_lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=args.lr_factor,
        patience=args.lr_patience,
        verbose=True
    )
    
    # 3. Setup dataloaders
    dataloaders = create_dataloaders(
        args,
        ref_only_df,
        cons_train_df,
        cons_val_df,
        label_encoder,
        torch_transform,
        'label',
        args.batch_size,
        #len(label_encoder.classes_),
        add_perspective=args.add_persp_aug
    )
    
    # 4. Setup Criterion (Loss)
    # Lưu ý: Nếu args không có domain_w, ta set mặc định
    domain_w = getattr(args, 'domain_w', 0.0)
    
    loss_weights = {
        'ce': args.ce_w,
        'arcface': args.arcface_w,
        'contrastive': args.contrastive_w,
        'triplet': args.triplet_w,
        'focal': args.focal_w,
        'domain': domain_w
    }
    
    print("Loss weights:", loss_weights)
    
    # Sử dụng EnhancedMultiheadLoss hoặc Criterion mặc định
    criterion = EnhancedMultiheadLoss(
        len(label_encoder.classes_),
        args.metric_margin,
        HardNegativePairSelector(),
        args.metric_margin,
        RandomNegativeTripletSelector(args.metric_margin),
        use_cosine=(args.metric_evaluator_type == 'cosine'),
        weights=loss_weights,
        focal_gamma=args.focal_gamma,
        use_side_labels=args.train_with_side_labels,
        use_domain_adaptation=getattr(args, 'use_domain_adaptation', False)
    )
    
    # 5. Train Model (Sử dụng hneg_train_model đã fix lỗi Assertion)
    model, val_metrics = hneg_train_model(
        model, optimizer, scheduler,
        device, dataloaders,
        save_path, label_encoder, criterion,
        num_epochs=args.max_epochs,
        earlystop_patience=3 * (args.lr_patience + 1),
        simul_sidepairs=args.metric_simul_sidepairs_eval,
        train_with_side_labels=args.train_with_side_labels,
        sidepairs_agg=args.sidepairs_agg,
        metric_evaluator_type=args.metric_evaluator_type,
        val_evaluator='metric',
        args=args # Quan trọng: Truyền args vào để hneg_train_model check model_type
    )
    # =================================================

    print('completed train()')
    print('val_metrics', val_metrics)

    run_metrics_list = log_metrics(val_metrics, 'val')
    predictions_dfs_list = []

    from sanitytest_eval import create_eval_dataloaders

    # Kiểm tra lại model type để lấy embedding model đúng
    is_single_stream = getattr(args, 'model_type', '') == 'single_stream'
    
    evaluator = MetricEmbeddingEvaluator(model, args.metric_simul_sidepairs_eval,
        sidepairs_agg_method=args.sidepairs_agg, metric_evaluator_type = args.metric_evaluator_type)

    logit_evaluator = LogitEvaluator(model, args.metric_simul_sidepairs_eval, sidepairs_agg_method=args.sidepairs_agg)

    def test_model(de_imgs_df, evaluator, dataset_name, run_metrics_list, predictions_dfs_list, rotate_aug=None):
        if rotate_aug is not None:
            dataset_name += "_rotate_aug{}".format(rotate_aug)

        print("Evaluating", dataset_name)
        eval_dataloader, eval_dataset = create_eval_dataloaders(
            de_imgs_df, label_encoder, torch_transform,
            'label', 24, rotate_aug = rotate_aug
        )

        ref_dataloader, _ = create_eval_dataloaders(
            ref_only_df, label_encoder, torch_transform,
            'label', 24, rotate_aug=rotate_aug
        )
        dataloader = {'ref':ref_dataloader, 'eval':eval_dataloader }

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Eval {}: {} images from {} total images".format(dataset_name, len(eval_dataset), len(de_imgs_df)))

        # Nếu là LogitEvaluator, cần gán multihead_model
        if isinstance(evaluator, LogitEvaluator):
            evaluator.multihead_model = model
        # Nếu là MetricEmbeddingEvaluator, gán embedding_model/backbone
        elif isinstance(evaluator, MetricEmbeddingEvaluator):
             evaluator.siamese_model = model.backbone if is_single_stream else model.embedding_model

        metrics_results, predictions = evaluator.eval_model(device, dataloader, do_pr_metrics=True, add_single_side_eval=True)

        if 'PR-curve' in metrics_results:
            plot_pr_curve(metrics_results['PR-curve'], dataset_name)

        run_metrics_list += log_metrics(metrics_results, dataset_name)

        predictions['dataset'] = dataset_name
        predictions['val_imgs_csv'] = os.path.split(args.val_imgs_csv)[1]
        predictions_dfs_list.append(predictions)

        return metrics_results, predictions

    test_model(test_df, logit_evaluator, 'holdout-logit', run_metrics_list, predictions_dfs_list)
    test_model(test_df, evaluator, 'holdout', run_metrics_list, predictions_dfs_list)

    run_metrics_df = pd.DataFrame(run_metrics_list, columns=['val_imgs_csv', 'dataset', 'name', 'value'])
    all_predictions_df = pd.concat(predictions_dfs_list, ignore_index = True)

    # Save results
    for target_save_dir in [save_path, 'outputs']:
        print(f'saving predictions {target_save_dir}')
        os.makedirs(target_save_dir, exist_ok=True)
        all_predictions_df.to_csv(os.path.join(target_save_dir, 'eval_predictions_{}'.format(os.path.basename(args.val_imgs_csv))))

    torch.save(model.state_dict(), os.path.join(save_path, '{}.pth'.format(os.path.basename(args.val_imgs_csv))))

    return run_metrics_df, all_predictions_df

if __name__ == '__main__':
    import arguments
    import os

    args = arguments.nocv_parser().parse_args()
    run_results, all_predictions_df = run(args)