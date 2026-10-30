import math
import torch
from torch import nn
from torch.nn import functional as F

class HSSGA(nn.Module):

    def __init__(self, input_dim=768, state_dim=64, gaussian_dim=8):
        super().__init__()
        self.state_dim = state_dim
        self.rho = 0.1
        self.mean_bound = 0.5
        self.log_precision_bound = math.log(2.0)
        self.input_projection = nn.Linear(3 * input_dim, state_dim)
        self.delta_projection = nn.Linear(state_dim, state_dim)
        self.b_projection = nn.Linear(state_dim, state_dim)
        self.c_projection = nn.Linear(state_dim, state_dim)
        self.attention = nn.Linear(state_dim, 1, bias=False)
        self.raw_decay = nn.Parameter(torch.full((state_dim,), math.log(math.expm1(0.9))))
        self.output = nn.Linear(state_dim, 2 * gaussian_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def padded_indices(rows, device, width=None):
        width = width or max(1, max((len(row) for row in rows), default=0))
        indices = torch.tensor([row + [0] * (width - len(row)) for row in rows], dtype=torch.long, device=device)
        lengths = torch.tensor([len(row) for row in rows], dtype=torch.long, device=device)
        mask = torch.arange(width, device=device)[None, :] < lengths[:, None]
        return (indices, mask)

    def propagate(self, features, layout):
        own = features[layout['job_nodes']]
        (parent_ids, mask) = self.padded_indices(layout['parent_nodes'], features.device)
        (sibling_ids, sibling_mask) = self.padded_indices(layout['sibling_nodes'], features.device)
        sibling_count = sibling_mask.sum(-1, keepdim=True)
        sibling_mean = (features[sibling_ids] * sibling_mask[..., None]).sum(1) / sibling_count.clamp_min(1)
        sibling_difference = (own - sibling_mean) * (sibling_count > 0)
        parent = features[parent_ids]
        expanded_own = own[:, None, :].expand_as(parent)
        expanded_sibling = sibling_difference[:, None, :].expand_as(parent)
        x = torch.tanh(self.input_projection(torch.cat((expanded_own, expanded_own - parent, expanded_sibling), -1)))
        delta = F.softplus(self.delta_projection(x)) + 0.0001
        decay = -(0.1 + F.softplus(self.raw_decay))
        transition = torch.exp(delta * decay)
        drive = torch.expm1(delta * decay) / decay * torch.tanh(self.b_projection(x)) * x
        readout = torch.tanh(self.c_projection(x))
        logits = self.attention(x).squeeze(-1)
        weights = logits.masked_fill(~mask, torch.finfo(logits.dtype).min).softmax(-1) * mask
        first_states = (weights[..., None] * drive).sum(1)
        anchors = layout['anchor_jobs']
        (parent_jobs, parent_mask) = self.padded_indices(layout['anchor_parent_jobs'], features.device, width=parent_ids.shape[1])
        second_edges = transition[anchors] * first_states[parent_jobs] + drive[anchors]
        context = (weights[anchors, :, None] * readout[anchors] * second_edges).sum(1)
        available = parent_mask.any(-1)
        retention = (weights[anchors] * transition[anchors].mean(-1)).sum(-1)
        return (context, available, retention)

    def fuse(self, center, offset, context, available):
        (mean_raw, precision_raw) = self.output(context).chunk(2, -1)
        precision_ratio = torch.exp(self.log_precision_bound * torch.tanh(precision_raw))
        rho = self.rho * available.to(center.dtype)[:, None]
        precision_multiplier = 1 + rho * (precision_ratio - 1)
        weight = rho * precision_ratio / precision_multiplier
        standardized_shift = weight * self.mean_bound * torch.tanh(mean_raw)
        std_ratio = precision_multiplier.rsqrt()
        enhanced_center = center + offset * standardized_shift
        enhanced_offset = offset * std_ratio
        precision_ratio = torch.where(available[:, None], precision_ratio, torch.ones_like(precision_ratio))
        information = {'context_available': available, 'precision_ratio': precision_ratio, 'effective_weight': weight, 'standardized_mean_shift': standardized_shift, 'std_ratio': std_ratio, 'context_norm': context.norm(dim=-1)}
        return ((enhanced_center, enhanced_offset), information)

    def forward(self, features, layout, center, offset):
        (context, available, retention) = self.propagate(features, layout)
        (box, information) = self.fuse(center, offset, context, available)
        information['retention'] = retention
        return (box, information)
