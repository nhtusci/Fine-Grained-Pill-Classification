import torch
import torch.nn as nn
from .margin_linear import MarginLinear, l2_norm
import torch.nn.functional as F

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

def grad_reverse(x, alpha=1.0):
    return GradientReversal.apply(x, alpha)

class DomainHead(nn.Module):
    def __init__(self, emb_size, hidden_dim=256, dropout=0.5):
        super(DomainHead, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(emb_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2) # 2 lớp: Reference và Consumer
        )

    def forward(self, x, alpha=1.0):
        x = grad_reverse(x, alpha) # Đảo ngược gradient khi quay về backbone
        return self.fc(x)

class MultiheadModel(nn.Module):
    def __init__(self, embedding_model, n_classes, train_with_side_labels=True):
        super(MultiheadModel, self).__init__()

        self.embedding_model = embedding_model
        if train_with_side_labels:
            n_classes *= 2
            print(f"treat front/back as different classes (first half: front, second half: back), n_classes={n_classes}") 
        self.n_classes = n_classes
        emb_size = embedding_model.out_features

        self.binary_head = BinaryHead(n_classes, emb_size)
        self.margin_head = MarginHead(n_classes, emb_size)
        
        # THÊM DÒNG NÀY: Khởi tạo domain head
        self.domain_head = DomainHead(emb_size)

        self.train_with_side_labels = train_with_side_labels

    def forward(self, x, target, **kwargs):
        # 1. Lấy alpha ra từ kwargs, nếu không có thì mặc định 1.0
        # Dùng .pop() để lấy giá trị ra VÀ xóa luôn nó khỏi kwargs
        alpha = kwargs.pop('alpha', 1.0)

        # the multi-instance model requires index array argument
        emb = self.embedding_model(x, **kwargs)
        logits = self.binary_head(emb)

        # Tính domain logits
        domain_logits = self.domain_head(emb, alpha)

        if target is None:
            # for get_embedding
            return {'emb': emb, 'logits': logits, 'domain_logits': domain_logits} # Thêm vào đây
        
        arcface_logits = self.margin_head(emb, target, is_infer=False)

        return {'emb': emb, 'logits': logits, 'arcface_logits': arcface_logits, 'domain_logits': domain_logits} # Thêm vào đây}

    def get_embedding(self, x, **kwargs):
        return self.forward(x, None, **kwargs)['emb']

    def shift_label_indexes(self, logits):
        assert self.train_with_side_labels
        actual_n_classes = self.n_classes // 2
        f = logits[:, :actual_n_classes]
        b = logits[:, actual_n_classes:]
        assert f.shape == b.shape        
        logits = torch.stack([f, b], dim=0)
        logits, _ = logits.max(dim=0)

        return logits

    def get_original_n_classes(self):
        if self.train_with_side_labels:
            return self.n_classes // 2
        else:
            return self.n_classes

    def get_original_logits(self, x, softmax=False, **kwargs):
        logits = self.forward(x, None, **kwargs)['logits']
        if softmax:
            logits = F.softmax(logits, dim=1)

        if self.train_with_side_labels:
            # shift back the logits to the original classes
            logits = self.shift_label_indexes(logits)

        return logits


class BinaryHead(nn.Module):
    def __init__(self, num_class=1000, emb_size=512, s=64.0):
        super(BinaryHead, self).__init__()
        self.s = s
        self.fc = nn.Sequential(nn.Linear(emb_size, num_class))

    def forward(self, fea):
        fea = l2_norm(fea)
        logits = self.fc(fea) * self.s

        return logits


class MarginHead(nn.Module):
    def __init__(self, num_class=1000, emb_size=512, s=64., m=0.5):
        super(MarginHead, self).__init__()
        self.fc = MarginLinear(embedding_size=emb_size, classnum=num_class, s=s, m=m)

    def forward(self, fea, label, is_infer):
        fea = l2_norm(fea)
        logits = self.fc(fea, label, is_infer)

        return logits
