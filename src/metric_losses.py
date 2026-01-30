import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all', **kwargs):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size, device=device).view(-1, 1), 0
        )
        mask = mask * logits_mask
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        pos_counts = mask.sum(1)
        non_zero = pos_counts > 0
        mean_log_prob_pos = torch.zeros_like(pos_counts, device=device)
        if non_zero.any():
            mean_log_prob_pos[non_zero] = (mask[non_zero] * log_prob[non_zero]).sum(1) / (pos_counts[non_zero] + 1e-8)
        if non_zero.any():
            loss = -mean_log_prob_pos[non_zero].mean()
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
        return loss


class TripletLoss(nn.Module):
    def __init__(self, margin=0.3, mining='hard', **kwargs):
        super().__init__()
        self.margin = margin
        self.mining = mining

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, dim=1)
        pairwise_dist = torch.cdist(embeddings, embeddings, p=2)
        batch_size = embeddings.size(0)
        loss = 0.0
        count = 0
        for i in range(batch_size):
            anchor_label = labels[i]
            positive_mask = (labels == anchor_label) & (torch.arange(batch_size, device=labels.device) != i)
            negative_mask = labels != anchor_label
            if positive_mask.sum() == 0 or negative_mask.sum() == 0:
                continue
            if self.mining == 'hard':
                pos_dist = pairwise_dist[i][positive_mask].max()
                neg_dist = pairwise_dist[i][negative_mask].min()
            elif self.mining == 'semi_hard':
                pos_dist = pairwise_dist[i][positive_mask].mean()
                neg_candidates = pairwise_dist[i][negative_mask]
                semi_hard = neg_candidates[(neg_candidates > pos_dist) & (neg_candidates < pos_dist + self.margin)]
                neg_dist = semi_hard.min() if len(semi_hard) > 0 else neg_candidates.min()
            else:
                pos_dist = pairwise_dist[i][positive_mask].mean()
                neg_dist = pairwise_dist[i][negative_mask].mean()
            loss += F.relu(pos_dist - neg_dist + self.margin)
            count += 1
        return loss / max(count, 1)


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, device='cuda', **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))

    def forward(self, features, labels):
        centers_batch = self.centers[labels]
        return F.mse_loss(features, centers_batch)


class ArcFaceLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, scale=64.0, margin=0.5, device='cuda', **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.scale = scale
        self.margin = margin
        import math
        cos_m = math.cos(margin)
        sin_m = math.sin(margin)
        self.register_buffer('cos_m', torch.tensor(cos_m).to(device))
        self.register_buffer('sin_m', torch.tensor(sin_m).to(device))
        self.register_buffer('th', torch.tensor(math.cos(math.pi - margin)).to(device))
        self.register_buffer('mm', torch.tensor(math.sin(math.pi - margin) * margin).to(device))
        self.W = nn.Parameter(torch.FloatTensor(num_classes, feat_dim).to(device))
        nn.init.xavier_uniform_(self.W)

    def forward(self, features, labels):
        features = F.normalize(features, dim=1)
        W = F.normalize(self.W, dim=1)
        logits = torch.mm(features, W.t())
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = F.one_hot(labels, self.num_classes).float()
        target_logits = torch.cos(theta + self.margin)
        target_logits = torch.where(logits > self.th, target_logits, logits - self.mm)
        logits = logits * (1 - one_hot) + target_logits * one_hot
        logits *= self.scale
        return F.cross_entropy(logits, labels)


class CosFaceLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, scale=64.0, margin=0.35, device='cuda', **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.scale = scale
        self.margin = margin
        self.W = nn.Parameter(torch.FloatTensor(num_classes, feat_dim).to(device))
        nn.init.xavier_uniform_(self.W)

    def forward(self, features, labels):
        features = F.normalize(features, dim=1)
        W = F.normalize(self.W, dim=1)
        logits = torch.mm(features, W.t())
        one_hot = F.one_hot(labels, self.num_classes).float()
        logits = logits - one_hot * self.margin
        logits *= self.scale
        return F.cross_entropy(logits, labels)


def _optional_pytorch_metric_learning_loss(loss_name, num_classes, feat_dim, device, **kwargs):
    try:
        from pytorch_metric_learning import losses as pml_losses
    except ImportError:
        return None
    name_map = {
        'arcface_pml': ('ArcFaceLoss', {'num_classes': num_classes, 'embedding_size': feat_dim}),
        'cosface_pml': ('CosFaceLoss', {'num_classes': num_classes, 'embedding_size': feat_dim}),
        'sphereface': ('SphereFaceLoss', {'num_classes': num_classes, 'embedding_size': feat_dim}),
        'large_margin_softmax': ('LargeMarginSoftmaxLoss', {'num_classes': num_classes, 'embedding_size': feat_dim}),
    }
    if loss_name not in name_map:
        return None
    class_name, base_kw = name_map[loss_name]
    base_kw.update({k: v for k, v in kwargs.items() if k != 'name' and k != 'weight'})
    try:
        cls = getattr(pml_losses, class_name)
        return cls(**base_kw).to(device)
    except Exception:
        return None


_COMPONENT_REGISTRY = {
    'contrastive': SupervisedContrastiveLoss,
    'supcon': SupervisedContrastiveLoss,
    'triplet': TripletLoss,
    'center': CenterLoss,
    'arcface': ArcFaceLoss,
    'cosface': CosFaceLoss,
}


def build_metric_loss_from_config(metric_config, num_classes, feat_dim, device='cuda'):
    num_classes = getattr(metric_config, 'num_classes', None) or num_classes
    feat_dim = getattr(metric_config, 'feat_dim', None) or feat_dim
    components_list = getattr(metric_config, 'components', None)
    if not components_list:
        components_list = [
            {'name': 'center', 'weight': 0.1},
            {'name': 'triplet', 'weight': 0.5, 'margin': 0.3, 'mining': 'hard'},
        ]
    loss = ComposableMetricLoss(
        num_classes=num_classes,
        feat_dim=feat_dim,
        device=device,
        component_specs=components_list,
    )
    return loss


class ComposableMetricLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, device='cuda', component_specs=None):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        self.component_specs = component_specs or []
        self._components = nn.ModuleList()
        self._weights = []

        for spec in self.component_specs:
            if hasattr(spec, 'name'):
                name = getattr(spec, 'name')
                weight = getattr(spec, 'weight', 1.0)
                kwargs = {k: getattr(spec, k) for k in dir(spec) if not k.startswith('_') and k not in ('name', 'weight')}
            else:
                name = spec.get('name')
                weight = spec.get('weight', 1.0)
                kwargs = {k: v for k, v in spec.items() if k not in ('name', 'weight')}

            if not name:
                continue
            name_lower = name.lower().strip()
            if name_lower in _COMPONENT_REGISTRY:
                cls = _COMPONENT_REGISTRY[name_lower]
                c_kw = dict(kwargs)
                if cls in (CenterLoss, ArcFaceLoss, CosFaceLoss):
                    c_kw.setdefault('num_classes', num_classes)
                    c_kw.setdefault('feat_dim', feat_dim)
                    c_kw.setdefault('device', device)
                try:
                    self._components.append(cls(**c_kw))
                    self._weights.append(float(weight))
                except Exception as e:
                    raise RuntimeError(f"Failed to build metric component '{name}': {e}") from e
            else:
                ext = _optional_pytorch_metric_learning_loss(name_lower, num_classes, feat_dim, device, **kwargs)
                if ext is not None:
                    self._components.append(ext)
                    self._weights.append(float(weight))
                else:
                    raise ValueError(
                        f"Unknown metric component: '{name}'. "
                        f"Available: {list(_COMPONENT_REGISTRY.keys())}"
                    )

        if not self._components:
            raise ValueError("metric_loss.components must contain at least one known component.")

    def forward(self, features, labels):
        loss_dict = {}
        total = 0.0
        for i, (module, w) in enumerate(zip(self._components, self._weights)):
            loss_val = module(features, labels)
            key = getattr(module, '__component_name__', None) or f"comp_{i}"
            if hasattr(module, '__class__'):
                key = module.__class__.__name__.replace('Loss', '').lower()
                if key == 'supervisedcontrastive':
                    key = 'contrastive'
            loss_dict[key] = loss_val.item() if isinstance(loss_val, torch.Tensor) else float(loss_val)
            total = total + w * loss_val
        loss_dict['total_metric'] = total.item() if isinstance(total, torch.Tensor) else float(total)
        return total, loss_dict


class ClusterLoss(ComposableMetricLoss):
    def __init__(self, num_classes, feat_dim, temperature=0.07, use_triplet=False, triplet_margin=0.3,
                 weight_contrastive=1.0, weight_center=0.1, weight_triplet=0.5, device='cuda'):
        components = [
            {'name': 'contrastive', 'weight': weight_contrastive, 'temperature': temperature},
            {'name': 'center', 'weight': weight_center},
        ]
        if use_triplet:
            components.append({'name': 'triplet', 'weight': weight_triplet, 'margin': triplet_margin})
        super().__init__(
            num_classes=num_classes,
            feat_dim=feat_dim,
            device=device,
            component_specs=components,
        )
