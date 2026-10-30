import os
import logging
import pickle as pkl
import torch
import numpy as np
import sys
import torch.nn as nn
from baseline_layers import MLP, MLP_Large
from transformers import BertModel, AutoModel
import torch.nn.functional as F
logging.getLogger('transformers').setLevel(logging.ERROR)

def backbone_path(args):
    return str(args.model_path)

class GaussianBox(nn.Module):

    def __init__(self, args):
        super(GaussianBox, self).__init__()
        self.args = args
        self.pre_train_model = self.__load_pre_trained__()
        self.par_projection_center = MLP(input_dim=768, hidden=self.args.hidden, output_dim=self.args.embed_size, dropout=self.args.dropout)
        self.par_projection_delta = MLP(input_dim=768, hidden=self.args.hidden, output_dim=self.args.embed_size, dropout=self.args.dropout)
        self.child_projection_center = MLP(input_dim=768, hidden=self.args.hidden, output_dim=self.args.embed_size, dropout=self.args.dropout)
        self.child_projection_delta = MLP(input_dim=768, hidden=self.args.hidden, output_dim=self.args.embed_size, dropout=self.args.dropout)
        self.gate_par_delta = MLP(input_dim=768, hidden=self.args.hidden, output_dim=self.args.embed_size, dropout=self.args.dropout)
        self.gate_child_delta = MLP(input_dim=768, hidden=self.args.hidden, output_dim=self.args.embed_size, dropout=self.args.dropout)
        self.parent_small_offset = MLP(input_dim=768, hidden=self.args.hidden, output_dim=self.args.embed_size, dropout=self.args.dropout)
        self.parent_large_offset = MLP_Large(input_dim=768, hidden=self.args.hidden, output_dim=self.args.embed_size, dropout=self.args.dropout)
        self.child_small_offset = MLP(input_dim=768, hidden=self.args.hidden, output_dim=self.args.embed_size, dropout=self.args.dropout)
        self.child_large_offset = MLP_Large(input_dim=768, hidden=self.args.hidden, output_dim=self.args.embed_size, dropout=self.args.dropout)
        self.parent_mlp = MLP(input_dim=768, hidden=self.args.hidden, output_dim=self.args.embed_size, dropout=self.args.dropout)
        self.child_mlp = MLP(input_dim=768, output_dim=self.args.embed_size, hidden=self.args.hidden, dropout=self.args.dropout)

    def __load_pre_trained__(self):
        if self.args.model == 'bert':
            model = BertModel.from_pretrained(backbone_path(self.args), local_files_only=True)
        elif self.args.model == 'snowflake':
            model = AutoModel.from_pretrained(backbone_path(self.args), add_pooling_layer=False, local_files_only=bool(self.args.model_path))
        elif self.args.model == 'e5':
            model = AutoModel.from_pretrained(backbone_path(self.args), local_files_only=True)
        return model

    def get_cls(self, encode_inputs):
        if self.args.model == 'snowflake':
            cls_embed = self.pre_train_model(**encode_inputs).last_hidden_state[:, 0]
        elif self.args.model == 'bert':
            outputs = self.pre_train_model(**encode_inputs)
            last_hidden_state = outputs.last_hidden_state
            cls_embed = last_hidden_state[:, 0, :]
        elif self.args.model == 'e5':
            outputs = self.pre_train_model(**encode_inputs)
            last_hidden_states = outputs.last_hidden_state
            attention_mask = encode_inputs['attention_mask']
            last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
            cls_embed = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        return cls_embed

    def par_projection_box(self, encode_inputs):
        return self.child_projection_box(encode_inputs)

    def child_projection_box(self, encode_inputs):
        embedding = self.get_cls(encode_inputs)
        center = self.child_projection_center(embedding)
        if self.args.method == 'normal':
            raw_offset = self.child_projection_delta(embedding)
        elif self.args.method == 'gated':
            gate = torch.sigmoid(self.gate_child_delta(embedding))
            small = self.child_small_offset(embedding)
            large = self.child_large_offset(embedding)
            raw_offset = gate * large + (1 - gate) * small
        else:
            raise ValueError(f'Unknown projection method: {self.args.method}')
        offset = torch.clamp(torch.exp(raw_offset), min=1e-38)
        return (center, offset)

    def mean_regularization(self, mu):
        return torch.norm(mu, dim=1).mean()

    def clip_covariance(self, sigma):
        variance = torch.diagonal(sigma, dim1=-2, dim2=-1)
        return F.relu(variance - self.args.max_variance).mean()

    def minimum_volume_regularization(self, sigma):
        variance = torch.diagonal(sigma, dim1=-2, dim2=-1)
        return torch.square(F.relu(self.args.min_variance - variance)).mean()

    def compute_gaussian(self, center, offset):
        mu = center
        sigma = torch.diag_embed(offset ** 2)
        return (mu, sigma)

    def bhattacharyya_distance(self, mu1, sigma1, mu2, sigma2):
        variance1 = self._diagonal_variance(sigma1)
        variance2 = self._diagonal_variance(sigma2)
        average = 0.5 * (variance1 + variance2)
        mean_term = 0.125 * torch.sum(torch.square(mu1 - mu2) / average, dim=-1)
        determinant_term = 0.5 * torch.sum(torch.log(average) - 0.5 * (torch.log(variance1) + torch.log(variance2)), dim=-1)
        return torch.clamp(mean_term + determinant_term, min=0.0)

    def bhattacharyya_coefficient(self, mu1, sigma1, mu2, sigma2):
        return torch.exp(-self.bhattacharyya_distance(mu1, sigma1, mu2, sigma2))

    def stochastic_containment_loss(self, mu_c: torch.Tensor, cov_c: torch.Tensor, mu_p: torch.Tensor, cov_p: torch.Tensor):
        sigma_p_inv = torch.linalg.inv(cov_p)
        trace_term = torch.einsum('bij,bij->b', sigma_p_inv, cov_c)
        mean_term = (mu_p - mu_c).unsqueeze(-1)
        mahalanbois = torch.matmul(torch.matmul(mean_term.transpose(1, 2), sigma_p_inv), mean_term).squeeze()
        return trace_term + mahalanbois

    def kl_divergence_gaussians(self, mu_c, sigma_c, mu_p, sigma_p):
        variance_c = self._diagonal_variance(sigma_c)
        variance_p = self._diagonal_variance(sigma_p)
        terms = (variance_c + torch.square(mu_p - mu_c)) / variance_p - 1.0 + torch.log(variance_p) - torch.log(variance_c)
        return torch.clamp(0.5 * torch.sum(terms, dim=-1), min=0.0)

    def child_volume(self, sigma_c, eps=1e-06):
        d = sigma_c.size(-1)
        eye = torch.eye(d, device=sigma_c.device).unsqueeze(0)
        sigma_stable = sigma_c + eps * eye
        log_det = torch.linalg.slogdet(sigma_stable)[1]
        volume = torch.exp(0.5 * log_det)
        return volume

    def log_det_term(self, sigma):
        return torch.sum(torch.log(self._diagonal_variance(sigma)), dim=-1)

    def intersection_distribution(self, sigma_c, sigma_p, mu_c, mu_p):
        sigma_c_inv = torch.linalg.inv(sigma_c)
        sigma_p_inv = torch.linalg.inv(sigma_p)
        mu_c = mu_c.view(mu_c.size(0), mu_c.size(1), 1)
        mu_p = mu_p.view(mu_p.size(0), mu_p.size(1), 1)
        intersection_sigma = torch.linalg.inv(torch.add(sigma_c_inv, sigma_p_inv))
        intersection_mean = intersection_sigma @ (sigma_c_inv @ mu_c + sigma_p_inv @ mu_p)
        intersection_mean = intersection_mean.view(intersection_mean.size(0), intersection_mean.size(1))
        return (intersection_mean, intersection_sigma)

    def forward(self, encode_parent, encode_child, encode_negative_parents, flag='trace'):
        (par_center, par_offset) = self.par_projection_box(encode_parent)
        (par_mu, par_sigma) = self.compute_gaussian(par_center, par_offset)
        (child_center, child_offset) = self.child_projection_box(encode_child)
        (child_mu, child_sigma) = self.compute_gaussian(child_center, child_offset)
        (neg_center, neg_offset) = self.par_projection_box(encode_negative_parents)
        (neg_mu, neg_sigma) = self.compute_gaussian(neg_center, neg_offset)
        pos_kld = self.kl_divergence_gaussians(child_mu, child_sigma, par_mu, par_sigma)
        reverse_kld = self.kl_divergence_gaussians(par_mu, par_sigma, child_mu, child_sigma)
        neg_kld = self.kl_divergence_gaussians(child_mu, child_sigma, neg_mu, neg_sigma)
        log_volume_gap = 0.5 * (self.log_det_term(par_sigma) - self.log_det_term(child_sigma))
        kl_loss = self.asymmetric_loss(pos_kld, neg_kld, reverse_kld, log_volume_gap).mean()
        pos_distance = self.bhattacharyya_distance(par_mu, par_sigma, child_mu, child_sigma)
        neg_distance = self.bhattacharyya_distance(neg_mu, neg_sigma, child_mu, child_sigma)
        non_overlap = torch.clamp(-torch.expm1(-neg_distance), min=torch.finfo(neg_distance.dtype).eps)
        bc_loss = (pos_distance - torch.log(non_overlap)).mean()
        regularization = sum((self.minimum_volume_regularization(covariance) + self.clip_covariance(covariance) for covariance in (par_sigma, child_sigma, neg_sigma)))
        return self.args.wtbce * bc_loss + self.args.wtkl * kl_loss + self.args.wtreg * regularization

    def _diagonal_variance(self, sigma):
        return torch.clamp(torch.diagonal(sigma, dim1=-2, dim2=-1), min=1e-12)

    def asymmetric_loss(self, pos_kld, neg_kld, reverse_kld, log_volume_gap):
        alignment = F.relu(pos_kld - neg_kld + self.args.margin)
        coverage = F.relu(self.args.C * log_volume_gap - reverse_kld)
        return alignment + self.args.lam * coverage
