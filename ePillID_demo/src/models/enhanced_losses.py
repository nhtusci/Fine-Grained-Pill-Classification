"""
Enhanced Loss Functions với Domain Adaptation Loss
Tích hợp vào MultiheadLoss hiện tại
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

try:
    from models.losses import MultiheadLoss, OnlineContrastiveLoss, OnlineTripletLoss
except ImportError:
    pass


class EnhancedMultiheadLoss(nn.Module):
    """
    Extended MultiheadLoss với Domain Adaptation Loss
    """
    def __init__(self, 
                 n_pilltypes, 
                 contrastive_margin, 
                 pair_selector, 
                 triplet_margin, 
                 triplet_selector, 
                 use_cosine=False, 
                 use_side_labels=True,
                 weights={'ce': 1.0, 'arcface': 1.0, 'contrastive': 1.0, 
                         'triplet': 1.0, 'focal': 0.0, 'domain': 0.1},
                 focal_gamma=0.0,
                 use_domain_adaptation=True):
        """
        Args:
            weights: dictionary với thêm 'domain' weight
            use_domain_adaptation: enable domain loss
        """
        super(EnhancedMultiheadLoss, self).__init__()
        
        self.n_pilltypes = n_pilltypes
        self.use_side_labels = use_side_labels
        self.weights = weights
        self.use_domain_adaptation = use_domain_adaptation
        
        # Original losses
        self.contrastive_loss = OnlineContrastiveLoss(
            contrastive_margin, pair_selector, use_cosine
        )
        self.triplet_loss = OnlineTripletLoss(
            triplet_margin, triplet_selector, use_cosine
        )
        
        if weights['focal'] > 0.0:
            from models.focal_loss import FocalLossWithOutOneHot
            print(f"Using focal loss with gamma={focal_gamma}, weight={weights['focal']}")
            self.focal_loss = FocalLossWithOutOneHot(gamma=focal_gamma)
        
        # Domain loss
        if self.use_domain_adaptation and weights.get('domain', 0.0) > 0.0:
            self.domain_criterion = nn.CrossEntropyLoss()
            print(f"Using domain adaptation loss with weight={weights['domain']}")
    
    def forward(self, outputs, target, is_front=None, is_ref=None, domain_logits=None):
        """
        Args:
            outputs: dict {'emb', 'logits', 'arcface_logits'}
            target: pill labels
            is_front: front/back labels
            is_ref: reference/consumer labels (for domain)
            domain_logits: domain predictions (batch_size, 2)
        
        Returns:
            losses dictionary
        """
        if not self.use_side_labels:
            is_front = None
        
        emb = outputs['emb']
        if emb.is_cuda:
            device = emb.get_device()
        else:
            device = torch.device('cpu')
        
        losses = {}
        weighted_loss = torch.zeros(1, dtype=torch.float).to(device)
        
        # ==================== Metric Losses ====================
        if self.weights['contrastive'] > 0.0:
            contrastive = self.contrastive_loss(emb, target, is_front=is_front, is_ref=is_ref)
            if contrastive is not None:
                contrastive['contrastive'] = contrastive.pop('loss')
                losses.update(contrastive)
                weighted_loss += contrastive['contrastive'] * self.weights['contrastive']
        
        if self.weights['triplet'] > 0.0:
            triplet = self.triplet_loss(emb, target, is_front=is_front, is_ref=is_ref)
            if triplet is not None:
                triplet['triplet'] = triplet.pop('loss')
                losses.update(triplet)
                weighted_loss += triplet['triplet'] * self.weights['triplet']
        
        losses['metric_loss'] = weighted_loss.clone().detach()
        
        # ==================== Classification Losses ====================
        if is_front is not None:
            target = target.clone().detach()
            target[~(is_front.bool())] += self.n_pilltypes
        
        if self.weights['ce'] > 0.0:
            losses['ce'] = F.cross_entropy(outputs['logits'], target, reduction='mean')
            weighted_loss += losses['ce'] * self.weights['ce']
        
        if self.weights['focal'] > 0.0:
            losses['focal'] = self.focal_loss(outputs['logits'], target)
            weighted_loss += losses['focal'] * self.weights['focal']
        
        if self.weights['arcface'] > 0.0:
            losses['arcface'] = F.cross_entropy(
                outputs['arcface_logits'], target, reduction='mean'
            )
            weighted_loss += losses['arcface'] * self.weights['arcface']
        
        # ==================== Domain Adaptation Loss ====================
        if self.use_domain_adaptation and self.weights.get('domain', 0.0) > 0.0:
            if domain_logits is not None and is_ref is not None:
                # Tạo domain labels: 0 = reference, 1 = consumer
                domain_labels = (~is_ref).long().to(device)
                
                # Domain classification loss
                domain_loss = self.domain_criterion(domain_logits, domain_labels)
                losses['domain'] = domain_loss
                weighted_loss += domain_loss * self.weights['domain']
                
                # Tính accuracy cho monitoring
                domain_preds = domain_logits.argmax(dim=1)
                domain_acc = (domain_preds == domain_labels).float().mean()
                losses['domain_acc'] = domain_acc.detach()
            else:
                warnings.warn("Domain adaptation enabled but domain_logits or is_ref is None")
        
        losses['loss'] = weighted_loss
        
        return losses


if __name__ == '__main__':
    from metric_utils import HardNegativePairSelector, RandomNegativeTripletSelector
    import numpy as np
    
    print("Testing Enhanced Multihead Loss...")
    
    # Setup
    D_in = 128
    n_classes = 10
    batch_size = 16
    
    labels = torch.from_numpy(np.array([0] * 8 + [1] * 8))
    is_ref = torch.from_numpy(np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]))
    is_front = torch.from_numpy(np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]))
    
    embeddings = torch.randn(batch_size, D_in)
    logits = torch.randn(batch_size, n_classes * 2)
    arcface_logits = torch.randn(batch_size, n_classes * 2)
    domain_logits = torch.randn(batch_size, 2)
    
    outputs = {
        'emb': embeddings,
        'logits': logits,
        'arcface_logits': arcface_logits
    }
    
    # Create loss
    weights = {
        'ce': 1.0,
        'arcface': 0.1,
        'contrastive': 1.0,
        'triplet': 1.0,
        'focal': 0.0,
        'domain': 0.5  # Domain weight
    }
    
    criterion = EnhancedMultiheadLoss(
        n_pilltypes=n_classes,
        contrastive_margin=1.0,
        pair_selector=HardNegativePairSelector(),
        triplet_margin=1.0,
        triplet_selector=RandomNegativeTripletSelector(1.0),
        use_cosine=False,
        use_side_labels=True,
        weights=weights,
        use_domain_adaptation=True
    )
    
    # Forward
    losses = criterion(outputs, labels, is_front=is_front, is_ref=is_ref, 
                      domain_logits=domain_logits)
    
    print("\nLoss values:")
    for k, v in losses.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.item():.4f}")
    
    # Check required keys
    required_keys = ['loss', 'ce', 'contrastive', 'triplet', 'domain', 'domain_acc']
    for key in required_keys:
        assert key in losses, f"Missing key: {key}"
    
    # Test backward
    losses['loss'].backward()
    assert embeddings.grad is not None, "Gradient should exist!"
    
    print("\n All tests passed!")