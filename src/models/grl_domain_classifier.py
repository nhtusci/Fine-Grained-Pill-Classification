"""
Gradient Reversal Layer và Domain Classifier
Paper: Unsupervised Domain Adaptation by Backpropagation (ICML 2015)
Xóa bỏ khoảng cách giữa reference và consumer images
"""
import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer
    Forward: pass qua không đổi
    Backward: đảo dấu gradient với coefficient lambda
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambda_
        return output, None


class GradientReversalLayer(nn.Module):
    """Wrapper cho GRL"""
    def __init__(self, lambda_=1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = lambda_
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
    
    def set_lambda(self, lambda_):
        """Điều chỉnh lambda theo training progress"""
        self.lambda_ = lambda_


class DomainClassifier(nn.Module):
    """
    Domain Classifier để phân biệt reference vs consumer images
    Với GRL, network sẽ học features domain-invariant
    """
    def __init__(self, in_features, hidden_dim=256, dropout_p=0.5):
        super(DomainClassifier, self).__init__()
        
        self.grl = GradientReversalLayer(lambda_=1.0)
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            
            nn.Linear(hidden_dim // 2, 2)  # 2 domains: reference (0) vs consumer (1)
        )
        
    def forward(self, x):
        """
        Args:
            x: embeddings từ backbone (batch_size, in_features)
        Returns:
            domain logits (batch_size, 2)
        """
        x = self.grl(x)
        domain_logits = self.classifier(x)
        return domain_logits
    
    def set_lambda(self, lambda_):
        """Điều chỉnh lambda của GRL"""
        self.grl.set_lambda(lambda_)


def compute_lambda(epoch, max_epoch, gamma=10, power=0.75):
    """
    Tính lambda theo schedule từ paper DANN
    lambda tăng dần từ 0 -> 1 theo training progress
    
    Args:
        epoch: epoch hiện tại
        max_epoch: tổng số epoch
        gamma: tham số scale
        power: tham số power
    
    Returns:
        lambda value
    """
    p = float(epoch) / float(max_epoch)
    lambda_ = 2.0 / (1.0 + torch.exp(torch.tensor(-gamma * p))) - 1.0
    return lambda_.item()


if __name__ == '__main__':
    # Test GRL
    print("Testing Gradient Reversal Layer...")
    x = torch.randn(4, 512, requires_grad=True)
    grl = GradientReversalLayer(lambda_=1.0)
    y = grl(x)
    
    # Test forward
    assert torch.all(x == y), "Forward should be identity!"
    
    # Test backward
    loss = y.sum()
    loss.backward()
    assert x.grad is not None, "Gradient should exist!"
    print(f"Input gradient mean: {x.grad.mean().item():.4f}")
    print(" GRL test passed!")
    
    # Test Domain Classifier
    print("\nTesting Domain Classifier...")
    dc = DomainClassifier(in_features=512, hidden_dim=256)
    embeddings = torch.randn(8, 512)
    domain_logits = dc(embeddings)
    print(f"Domain logits shape: {domain_logits.shape}")
    assert domain_logits.shape == (8, 2), "Output shape mismatch!"
    print(" Domain Classifier test passed!")
    
    # Test lambda schedule
    print("\nTesting lambda schedule...")
    lambdas = [compute_lambda(e, 100) for e in range(0, 101, 10)]
    print(f"Lambda values: {[f'{l:.3f}' for l in lambdas]}")
    print(" Lambda schedule test passed!")