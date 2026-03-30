"""
Enhanced Embedding Model với Coordinate Attention và Domain Adaptation
Tích hợp vào pipeline ePillID
"""
import torch
import torch.nn as nn
import sys
import os

# Import modules from models/
try:
    from models.embedding_model import EmbeddingModel
    from models.coordinate_attention import CoordinateAttention
    from models.grl_domain_classifier import DomainClassifier, compute_lambda
except ImportError as e:
    # For testing - raise error so user knows imports failed
    print(f"Warning: Import failed: {e}")
    # Import with relative paths as fallback
    from coordinate_attention import CoordinateAttention
    from grl_domain_classifier import DomainClassifier, compute_lambda


class EnhancedEmbeddingModel(nn.Module):
    """
    Embedding Model với 2 cải tiến:
    1. Coordinate Attention sau backbone
    2. Domain Classifier với GRL
    """
    def __init__(self, 
                 network='resnet50', 
                 pooling='GAvP', 
                 dropout_p=0.5, 
                 cont_dims=2048, 
                 pretrained=True,
                 use_coord_attention=True,
                 use_domain_adaptation=True,
                 ca_reduction=32,
                 domain_hidden_dim=256,
                 domain_dropout=0.5):
        """
        Args:
            network: backbone architecture (resnet18, resnet50, etc.)
            pooling: pooling method (GAvP, MPNCOV, BCNN, CBP)
            dropout_p: dropout rate
            cont_dims: embedding dimension
            pretrained: use pretrained weights
            use_coord_attention: enable Coordinate Attention
            use_domain_adaptation: enable Domain Adaptation với GRL
            ca_reduction: reduction ratio cho CA
            domain_hidden_dim: hidden dim cho domain classifier
            domain_dropout: dropout cho domain classifier
        """
        super(EnhancedEmbeddingModel, self).__init__()
        
        # Import here to avoid circular dependency
        from models import fast_MPN_COV_wrapper
        
        # Base model (backbone + pooling)
        self.base_model = fast_MPN_COV_wrapper.get_model(
            arch=network,
            repr_agg=pooling,
            num_classes=cont_dims,
            dimension_reduction=cont_dims,
            pretrained=pretrained
        )
        
        self.out_features = cont_dims
        self.use_coord_attention = use_coord_attention
        self.use_domain_adaptation = use_domain_adaptation
        
        # Lấy số channels từ backbone để thêm CA
        # Với ResNet, features là Sequential, lấy output của layer cuối
        if hasattr(self.base_model, 'features'):
            # Tìm output channels của backbone
            if network.startswith('resnet'):
                # ResNet: lấy channels từ layer cuối
                if network in ['resnet18', 'resnet34']:
                    backbone_channels = 512
                else:  # resnet50, resnet101, resnet152
                    backbone_channels = 2048
            elif network.startswith('vgg'):
                backbone_channels = 512
            else:
                backbone_channels = 512  # default
                
            # Coordinate Attention module
            if self.use_coord_attention:
                print(f"Adding Coordinate Attention with {backbone_channels} channels")
                self.coord_attention = CoordinateAttention(
                    in_channels=backbone_channels,
                    reduction=ca_reduction
                )
        
        # Domain Classifier với GRL
        if self.use_domain_adaptation:
            print(f"Adding Domain Classifier with GRL")
            self.domain_classifier = DomainClassifier(
                in_features=cont_dims,
                hidden_dim=domain_hidden_dim,
                dropout_p=domain_dropout
            )
        
        # Dropout và embedding layers (như original)
        self.dropout = nn.Dropout(p=dropout_p)
        self.emb = nn.Sequential(
            nn.Linear(cont_dims, 1000),
            nn.BatchNorm1d(1000, affine=True),
            nn.ReLU(inplace=True),
            nn.Linear(1000, cont_dims),
            nn.Tanh()
        )
    
    def forward(self, x, return_domain_logits=False):
        """
        Args:
            x: input images (batch_size, 3, H, W)
            return_domain_logits: return domain predictions (for training)
        
        Returns:
            embeddings hoặc (embeddings, domain_logits)
        """
        # Extract features từ backbone
        if hasattr(self.base_model, 'features'):
            features = self.base_model.features(x)
            
            # Apply Coordinate Attention
            if self.use_coord_attention:
                features = self.coord_attention(features)
            
            # Apply pooling
            if hasattr(self.base_model, 'representation'):
                x = self.base_model.representation(features)
            else:
                x = features
            
            # Flatten
            x = x.view(x.size(0), -1)
            
            # Classifier (trong base_model nếu có)
            if hasattr(self.base_model, 'classifier'):
                # Không dùng classifier cuối, chỉ lấy features
                pass
        else:
            # Fallback: sử dụng base_model trực tiếp
            x = self.base_model(x)
        
        # Dropout và embedding
        x = self.dropout(x)

        if x.dim() != 2:
            x = x.view(x.size(0), -1)

        embeddings = self.emb(x)
        
        # Domain classification (nếu cần)
        if self.use_domain_adaptation and return_domain_logits:
            domain_logits = self.domain_classifier(embeddings)
            return embeddings, domain_logits
        
        return embeddings
    
    def get_embedding(self, x):
        """Wrapper cho compatibility"""
        return self.forward(x, return_domain_logits=False)
    
    def set_domain_lambda(self, lambda_):
        """Điều chỉnh lambda của GRL"""
        if self.use_domain_adaptation:
            self.domain_classifier.set_lambda(lambda_)


def create_enhanced_model(args, use_coord_attention=True, use_domain_adaptation=True):
    """
    Factory function để tạo enhanced model từ args
    Compatible với pipeline hiện tại
    
    Args:
        args: argparse arguments
        use_coord_attention: enable CA
        use_domain_adaptation: enable GRL
    
    Returns:
        EnhancedEmbeddingModel instance
    """
    model = EnhancedEmbeddingModel(
        network=args.appearance_network,
        pooling=args.pooling,
        dropout_p=args.dropout,
        cont_dims=args.metric_embedding_dim,
        pretrained=True,
        use_coord_attention=use_coord_attention,
        use_domain_adaptation=use_domain_adaptation,
        ca_reduction=getattr(args, 'ca_reduction', 32),
        domain_hidden_dim=getattr(args, 'domain_hidden_dim', 256),
        domain_dropout=getattr(args, 'domain_dropout', 0.5)
    )
    
    return model


if __name__ == '__main__':
    print("Testing Enhanced Embedding Model...")
    
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
    
    # Test model
    model = create_enhanced_model(args, use_coord_attention=True, use_domain_adaptation=True)
    
    # Forward pass
    x = torch.randn(4, 3, 224, 224)
    
    # Without domain logits
    emb = model(x, return_domain_logits=False)
    print(f"Embeddings shape: {emb.shape}")
    assert emb.shape == (4, 2048), "Embedding shape mismatch!"
    
    # With domain logits
    emb, domain_logits = model(x, return_domain_logits=True)
    print(f"Embeddings shape: {emb.shape}")
    print(f"Domain logits shape: {domain_logits.shape}")
    assert domain_logits.shape == (4, 2), "Domain logits shape mismatch!"
    
    print("✓ All tests passed!")