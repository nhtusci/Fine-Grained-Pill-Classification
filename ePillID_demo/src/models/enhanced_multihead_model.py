"""
Enhanced Multihead Model với Domain Adaptation
Wrapper cho EnhancedEmbeddingModel
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from models.margin_linear import MarginLinear, l2_norm
    from enhanced_embedding_model import EnhancedEmbeddingModel
except ImportError:
    pass


class EnhancedMultiheadModel(nn.Module):
    """
    Multihead Model với Enhanced Embedding (CA + GRL)
    """
    def __init__(self, 
                 embedding_model, 
                 n_classes, 
                 train_with_side_labels=True,
                 return_domain_logits=True):
        """
        Args:
            embedding_model: EnhancedEmbeddingModel instance
            n_classes: số lượng pill classes
            train_with_side_labels: treat front/back as different classes
            return_domain_logits: return domain predictions during training
        """
        super(EnhancedMultiheadModel, self).__init__()
        
        self.embedding_model = embedding_model
        self.return_domain_logits = return_domain_logits
        
        if train_with_side_labels:
            n_classes *= 2
            print(f"Treating front/back as different classes: n_classes={n_classes}")
        
        self.n_classes = n_classes
        self.train_with_side_labels = train_with_side_labels
        emb_size = embedding_model.out_features
        
        # Binary head và Margin head (như original)
        self.binary_head = BinaryHead(n_classes, emb_size)
        self.margin_head = MarginHead(n_classes, emb_size)
    
    def forward(self, x, target, **kwargs):
        """
        Args:
            x: input images
            target: pill labels
            **kwargs: additional arguments
        
        Returns:
            dict với emb, logits, arcface_logits, và domain_logits (nếu có)
        """
        # Get embeddings (và domain logits nếu training)
        if self.training and self.return_domain_logits and \
           hasattr(self.embedding_model, 'use_domain_adaptation') and \
           self.embedding_model.use_domain_adaptation:
            emb, domain_logits = self.embedding_model(x, return_domain_logits=True)
        else:
            emb = self.embedding_model(x, return_domain_logits=False)
            domain_logits = None
        
        # Classification heads
        logits = self.binary_head(emb)
        
        outputs = {'emb': emb, 'logits': logits}
        
        # ArcFace logits (nếu có target)
        if target is not None:
            arcface_logits = self.margin_head(emb, target, is_infer=False)
            outputs['arcface_logits'] = arcface_logits
        
        # Domain logits
        if domain_logits is not None:
            outputs['domain_logits'] = domain_logits
        
        return outputs
    
    def get_embedding(self, x, **kwargs):
        """Wrapper cho compatibility"""
        return self.embedding_model.get_embedding(x)
    
    def shift_label_indexes(self, logits):
        """Shift labels từ front/back về original classes"""
        assert self.train_with_side_labels
        actual_n_classes = self.n_classes // 2
        f = logits[:, :actual_n_classes]
        b = logits[:, actual_n_classes:]
        assert f.shape == b.shape
        logits = torch.stack([f, b], dim=0)
        logits, _ = logits.max(dim=0)
        return logits
    
    def get_original_n_classes(self):
        """Get số classes gốc (không tính front/back)"""
        if self.train_with_side_labels:
            return self.n_classes // 2
        else:
            return self.n_classes
    
    def get_original_logits(self, x, softmax=False, **kwargs):
        """Get logits cho original classes"""
        outputs = self.forward(x, None, **kwargs)
        logits = outputs['logits']
        
        if softmax:
            logits = F.softmax(logits, dim=1)
        
        if self.train_with_side_labels:
            logits = self.shift_label_indexes(logits)
        
        return logits
    
    def set_domain_lambda(self, lambda_):
        """Điều chỉnh lambda của GRL"""
        if hasattr(self.embedding_model, 'set_domain_lambda'):
            self.embedding_model.set_domain_lambda(lambda_)


class BinaryHead(nn.Module):
    """Binary classification head"""
    def __init__(self, num_class=1000, emb_size=512, s=64.0):
        super(BinaryHead, self).__init__()
        self.s = s
        self.fc = nn.Sequential(nn.Linear(emb_size, num_class))
    
    def forward(self, fea):
        fea = l2_norm(fea)
        logits = self.fc(fea) * self.s
        return logits


class MarginHead(nn.Module):
    """ArcFace margin head"""
    def __init__(self, num_class=1000, emb_size=512, s=64., m=0.5):
        super(MarginHead, self).__init__()
        self.fc = MarginLinear(embedding_size=emb_size, classnum=num_class, s=s, m=m)
    
    def forward(self, fea, label, is_infer):
        fea = l2_norm(fea)
        logits = self.fc(fea, label, is_infer)
        return logits


if __name__ == '__main__':
    print("Testing Enhanced Multihead Model...")
    
    # Giả lập args
    class Args:
        appearance_network = 'resnet50'
        pooling = 'GAvP'
        dropout = 0.5
        metric_embedding_dim = 2048
        ca_reduction = 32
        domain_hidden_dim = 256
        domain_dropout = 0.5
    
    args = Args()
    
    # Create models
    from enhanced_embedding_model import create_enhanced_model
    
    emb_model = create_enhanced_model(
        args, 
        use_coord_attention=True, 
        use_domain_adaptation=True
    )
    
    model = EnhancedMultiheadModel(
        emb_model, 
        n_classes=100, 
        train_with_side_labels=True,
        return_domain_logits=True
    )
    
    # Test forward
    x = torch.randn(8, 3, 224, 224)
    target = torch.randint(0, 100, (8,))
    
    model.train()
    outputs = model(x, target)
    
    print(f"Embeddings shape: {outputs['emb'].shape}")
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"ArcFace logits shape: {outputs['arcface_logits'].shape}")
    
    if 'domain_logits' in outputs:
        print(f"Domain logits shape: {outputs['domain_logits'].shape}")
        assert outputs['domain_logits'].shape == (8, 2), "Domain shape mismatch!"
    
    # Test evaluation mode
    model.eval()
    with torch.no_grad():
        emb = model.get_embedding(x)
        print(f"Eval embeddings shape: {emb.shape}")
    
    print("\n All tests passed!")