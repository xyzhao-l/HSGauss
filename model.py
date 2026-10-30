import torch
from torch.nn import functional as F
from baseline_model import GaussianBox
from hssga import HSSGA

class HSSGATaxoBell(GaussianBox):

    def __init__(self, args):
        super().__init__(args)
        if args.gradient_checkpointing:
            self.pre_train_model.gradient_checkpointing_enable()
        with torch.random.fork_rng(devices=[]):
            self.hssga = HSSGA(768, args.state_dim, args.embed_size)

    @property
    def bert(self):
        return self.pre_train_model

    def get_cls(self, inputs):
        if '_features' in inputs:
            return inputs['_features']
        return super().get_cls(inputs)

    def child_projection_box(self, inputs):
        if '_box' in inputs:
            return inputs['_box']
        return super().child_projection_box(inputs)

    def encode(self, tokens, node_ids, device, batch_size):
        parts = []
        for start in range(0, len(node_ids), batch_size):
            indices = torch.tensor(node_ids[start:start + batch_size], dtype=torch.long)
            inputs = {key: value.index_select(0, indices).to(device) for (key, value) in tokens.items()}
            parts.append(super().get_cls(inputs))
        return torch.cat(parts)

    def text_box(self, features):
        return super().child_projection_box({'_features': features})

    def box_state(self, box):
        (mu, covariance) = self.compute_gaussian(*box)
        return {'mu': mu, 'covariance': covariance}

    def text_state(self, features):
        return self.box_state(self.text_box(features))

    def candidate_boxes(self, features, layout, base_box=None):
        (center, offset) = self.text_box(features[layout['anchor_nodes']]) if base_box is None else base_box
        if self.args.disable_hssga:
            return ((center, offset), {})
        return self.hssga(features, layout, center, offset)

    def scores(self, query, candidate):
        return {'BC': self.bhattacharyya_coefficient(query['mu'], query['covariance'], candidate['mu'], candidate['covariance']), 'KL': -self.kl_divergence_gaussians(query['mu'], query['covariance'], candidate['mu'], candidate['covariance'])}

    def relation_margins(self, query_box, candidate_box, batch):
        (query_mu, query_covariance) = self.compute_gaussian(*query_box)
        (mu, covariance) = self.compute_gaussian(*candidate_box)
        (p, n) = (batch['parent_indices'], batch['negative_indices'])
        pos_distance = self.bhattacharyya_distance(mu[p], covariance[p], query_mu, query_covariance)
        neg_distance = self.bhattacharyya_distance(mu[n], covariance[n], query_mu, query_covariance)
        positive_kl = self.kl_divergence_gaussians(query_mu, query_covariance, mu[p], covariance[p])
        negative_kl = self.kl_divergence_gaussians(query_mu, query_covariance, mu[n], covariance[n])
        return {'BC': neg_distance - pos_distance, 'KL': negative_kl - positive_kl}

    @staticmethod
    def counterfactual_losses(factual, counterfactual, keep_ratio, margin):
        reference = factual['BC'].detach()
        bc = ((reference > 0).to(reference.dtype) * F.relu(keep_ratio * reference - counterfactual['BC'])).mean()
        kl = F.softplus(margin - torch.minimum(factual['KL'], counterfactual['KL'])).mean()
        return {'BC': bc, 'KL': kl}

    def forward(self, tokens, batch, device):
        features = self.encode(tokens, batch['node_ids'], device, self.args.encode_batch_size)
        base_box = self.text_box(features[batch['anchor_nodes']])
        (candidates, details) = self.candidate_boxes(features, batch, base_box)
        query = self.text_box(features[batch['query_indices']])
        parent = tuple((value[batch['parent_indices']] for value in candidates))
        negative = tuple((value[batch['negative_indices']] for value in candidates))
        base_loss = super().forward({'_box': parent}, {'_box': query}, {'_box': negative})
        loss = base_loss
        zero = base_loss.detach().new_zeros(())
        statistics = {'hssga_loss': base_loss.detach(), 'cf_added_loss': zero, 'cf_active_fraction': zero, 'cf_bc_keep_loss': zero, 'cf_kl_robust_loss': zero, 'cf_bc_correct_fraction': zero, 'cf_bc_violation_fraction': zero, 'cf_kl_worse_fraction': zero}
        for name in ('bc', 'kl'):
            for field in ('contribution', 'margin_factual', 'margin_counterfactual', 'margin_abs_change'):
                statistics[f'cf_{name}_{field}'] = zero
        cf_enabled = self.training and (not self.args.disable_hssga) and (self.args.cf_bc_weight > 0 or self.args.cf_kl_weight > 0) and (self.args.cf_drop_rate > 0)
        if cf_enabled:
            if 'cf_layout' not in batch:
                raise ValueError('Counterfactual training requires a paired context batch')
            active = torch.tensor(batch['cf_active'], dtype=torch.bool, device=features.device)
            statistics['cf_active_fraction'] = active.float().mean().detach()
            if active.any():
                (cf_candidates, _) = self.candidate_boxes(features, batch['cf_layout'], base_box)
                factual = {name: values[active] for (name, values) in self.relation_margins(query, candidates, batch).items()}
                counterfactual = {name: values[active] for (name, values) in self.relation_margins(query, cf_candidates, batch).items()}
                terms = self.counterfactual_losses(factual, counterfactual, self.args.bc_keep_ratio, self.args.margin)
                added_loss = self.args.cf_bc_weight * terms['BC'] + self.args.cf_kl_weight * terms['KL']
                loss = base_loss + added_loss
                reference = factual['BC'].detach()
                statistics.update(cf_added_loss=added_loss.detach(), cf_bc_keep_loss=terms['BC'].detach(), cf_kl_robust_loss=terms['KL'].detach(), cf_bc_correct_fraction=(reference > 0).float().mean().detach(), cf_bc_violation_fraction=((reference > 0) & (counterfactual['BC'] < self.args.bc_keep_ratio * reference)).float().mean().detach(), cf_kl_worse_fraction=(counterfactual['KL'] < factual['KL']).float().mean().detach())
                for (name, weight) in (('BC', self.args.cf_bc_weight), ('KL', self.args.cf_kl_weight)):
                    prefix = 'cf_' + name.lower() + '_'
                    statistics[prefix + 'contribution'] = (weight * terms[name]).detach()
                    statistics[prefix + 'margin_factual'] = factual[name].mean().detach()
                    statistics[prefix + 'margin_counterfactual'] = counterfactual[name].mean().detach()
                    statistics[prefix + 'margin_abs_change'] = (counterfactual[name] - factual[name]).abs().mean().detach()
        statistics['loss'] = loss.detach()
        if details:
            statistics['context_coverage'] = details['context_available'].float().mean().detach()
            statistics['mean_abs_shift'] = details['standardized_mean_shift'].abs().mean().detach()
        return (loss, statistics)
