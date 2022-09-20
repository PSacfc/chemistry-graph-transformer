# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from data import get_dataset
from lr import PolynomialDecayLR
import torch
import math
import torch.nn as nn
import pytorch_lightning as pl

from utils.flag import flag, flag_bounded

import numpy as np
import torch.nn.functional as F
def softplus_inverse(x):
    return x + np.log(-np.expm1(-x))

class RBFLayer(nn.Module):
    def __init__(self, K=64, cutoff=10, dtype=torch.float):
        super().__init__()
        self.cutoff = cutoff

        centers = torch.tensor(softplus_inverse(np.linspace(1.0, np.exp(-cutoff), K)), dtype=dtype)
        self.centers = nn.Parameter(F.softplus(centers))

        widths = torch.tensor([softplus_inverse(0.5 / ((1.0 - np.exp(-cutoff) / K)) ** 2)] * K, dtype=dtype)
        self.widths = nn.Parameter(F.softplus(widths))
    def cutoff_fn(self, D):
        x = D / self.cutoff
        x3, x4, x5 = torch.pow(x, 3.0), torch.pow(x, 4.0), torch.pow(x, 5.0)
        return torch.where(x < 1, 1-6*x5+15*x4-10*x3, torch.zeros_like(x))
    def forward(self, D):
        D = D.unsqueeze(-1)
        return self.cutoff_fn(D) * torch.exp(-self.widths*torch.pow((torch.exp(-D) - self.centers), 2))

class GraphFormer(pl.LightningModule):
    def __init__(
            self,
            n_layers,
            head_size,
            hidden_dim,
            dropout_rate,
            intput_dropout_rate,
            weight_decay,
            ffn_dim,
            dataset_name,
            warmup_updates,
            tot_updates,
            peak_lr,
            end_lr,
            edge_type,
            multi_hop_max_dist,
            attention_dropout_rate,
            flag=False,
            flag_m=3,
            flag_step_size=1e-3,
            flag_mag=1e-3,
        ):
        super().__init__()
        self.save_hyperparameters()

        self.head_size = head_size
        if dataset_name == 'ZINC':
            self.atom_encoder = nn.Embedding(64, hidden_dim, padding_idx=0)
            self.edge_encoder = nn.Embedding(64, head_size, padding_idx=0)
            self.edge_type = edge_type
            if self.edge_type == 'multi_hop':
                self.edge_dis_encoder = nn.Embedding(40 * head_size * head_size,1)
            self.rel_pos_encoder = nn.Embedding(40, head_size, padding_idx=0)
            self.in_degree_encoder = nn.Embedding(64, hidden_dim, padding_idx=0)
            self.out_degree_encoder = nn.Embedding(64, hidden_dim, padding_idx=0)
        else:
            self.atom_encoder = nn.Embedding(128 * 37 + 1, hidden_dim, padding_idx=0)
            self.edge_encoder = nn.Embedding(128 * 6 + 1, head_size, padding_idx=0)
            self.edge_type = edge_type
            if self.edge_type == 'multi_hop':
                self.edge_dis_encoder = nn.Embedding(128 * head_size * head_size,1)
            self.rel_pos_encoder = nn.Embedding(512, head_size, padding_idx=0)
            self.in_degree_encoder = nn.Embedding(512, hidden_dim, padding_idx=0)
            self.out_degree_encoder = nn.Embedding(512, hidden_dim, padding_idx=0)

        self.input_dropout = nn.Dropout(intput_dropout_rate)
        encoders = [EncoderLayer(hidden_dim, ffn_dim, dropout_rate, attention_dropout_rate, head_size, i)
                    for i in range(n_layers)]
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(hidden_dim)

        if dataset_name == 'PCQM4M-LSC':
            self.out_proj = nn.Linear(hidden_dim, 1)
        else:
            self.downstream_out_proj = nn.Linear(hidden_dim, get_dataset(dataset_name)['num_class'])

        self.graph_token = nn.Embedding(1, hidden_dim)
        self.graph_token_virtual_distance = nn.Embedding(1, head_size)

        self.evaluator = get_dataset(dataset_name)['evaluator']
        self.metric = get_dataset(dataset_name)['metric']
        self.loss_fn = get_dataset(dataset_name)['loss_fn']
        self.dataset_name = dataset_name
        
        self.warmup_updates = warmup_updates
        self.tot_updates = tot_updates
        self.peak_lr = peak_lr
        self.end_lr = end_lr
        self.weight_decay = weight_decay
        self.multi_hop_max_dist = multi_hop_max_dist

        self.flag = flag
        self.flag_m = flag_m
        self.flag_step_size = flag_step_size
        self.flag_mag = flag_mag
        self.hidden_dim = hidden_dim
        self.automatic_optimization = not self.flag

        K = 256
        cutoff = 10
        self.rbf = RBFLayer(K, cutoff)
        self.rel_pos_3d_proj = nn.Linear(K, head_size)

    def forward(self, batched_data, perturb=None):
        attn_bias, rel_pos, x = batched_data.attn_bias, batched_data.rel_pos, batched_data.x
        in_degree, out_degree = batched_data.in_degree, batched_data.in_degree
        edge_input, attn_edge_type = batched_data.edge_input, batched_data.attn_edge_type
        all_rel_pos_3d_1 = batched_data.all_rel_pos_3d_1
        bcc_shortest_dis = batched_data.bcc_shortest_dis

        # graph_attn_bias
        n_graph, n_node = x.size()[:2]
        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(1, self.head_size, 1, 1) # [n_graph, n_head, n_node+1, n_node+1]

        # rel pos
        # rel_pos_bias = self.rel_pos_encoder(rel_pos).permute(0, 3, 1, 2) # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        # graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + rel_pos_bias

        rbf_result = self.rel_pos_3d_proj(self.rbf(all_rel_pos_3d_1)).permute(0, 3, 1, 2)
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + rbf_result

        # reset rel pos here
        t = self.graph_token_virtual_distance.weight.view(1, self.head_size, 1)
        graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
        graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

        # edge feature
        if self.edge_type == 'multi_hop':
            rel_pos_ = rel_pos.clone()
            rel_pos_[rel_pos_ == 0] = 1 # set pad to 1
            rel_pos_ = torch.where(rel_pos_ > 1, rel_pos_ - 1, rel_pos_) # set 1 to 1, x > 1 to x - 1
            if self.multi_hop_max_dist > 0:
                rel_pos_ = rel_pos_.clamp(0, self.multi_hop_max_dist)
                edge_input = edge_input[:, :, :, :self.multi_hop_max_dist, :]
            edge_input = self.edge_encoder(edge_input).mean(-2) #[n_graph, n_node, n_node, max_dist, n_head]
            max_dist = edge_input.size(-2)
            edge_input_flat = edge_input.permute(3,0,1,2,4).reshape(max_dist, -1, self.head_size)
            edge_input_flat = torch.bmm(edge_input_flat, self.edge_dis_encoder.weight.reshape(-1, self.head_size, self.head_size)[:max_dist, :, :])
            edge_input = edge_input_flat.reshape(max_dist, n_graph, n_node, n_node, self.head_size).permute(1,2,3,0,4)
            edge_input = (edge_input.sum(-2) / (rel_pos_.float().unsqueeze(-1))).permute(0, 3, 1, 2)
        else:
            edge_input = self.edge_encoder(attn_edge_type).mean(-2).permute(0, 3, 1, 2) # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]

        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + edge_input
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1) # reset

        # node feauture + graph token
        node_feature = self.atom_encoder(x).mean(dim=-2)           # [n_graph, n_node, n_hidden]
        if self.flag and perturb is not None:
            node_feature += perturb

        node_feature = node_feature + self.in_degree_encoder(in_degree) + self.out_degree_encoder(out_degree)
        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)
        graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)

        # transfomrer encoder
        output = self.input_dropout(graph_node_feature)
        for enc_layer in self.layers:
            output = enc_layer(output, graph_attn_bias, rel_pos, bcc_shortest_dis)
        output = self.final_ln(output)

        # output part
        if self.dataset_name == 'PCQM4M-LSC':
            output = self.out_proj(output[:, 0, :])                        # get whole graph rep
        else:
            output = self.downstream_out_proj(output[:, 0, :])
        return output

    def training_step(self, batched_data, batch_idx):
        if self.dataset_name == 'ogbg-molpcba':
            if not self.flag:
                y_hat = self(batched_data).view(-1)
                y_gt = batched_data.y.view(-1).float()
                mask = ~torch.isnan(y_gt)
                loss = self.loss_fn(y_hat[mask], y_gt[mask])
            else:
                y_gt = batched_data.y.view(-1).float()
                mask = ~torch.isnan(y_gt)

                forward = lambda perturb: self(batched_data, perturb)
                model_forward = (self, forward)
                n_graph, n_node = batched_data.x.size()[:2]
                perturb_shape = (n_graph, n_node, self.hidden_dim)

                optimizer = self.optimizers()
                optimizer.zero_grad()
                loss, _ = flag(model_forward, perturb_shape, y_gt[mask], optimizer, batched_data.x.device, self.loss_fn,
                               m=self.flag_m, step_size=self.flag_step_size, mask=mask)

        elif self.dataset_name == 'ogbg-molhiv':
            if not self.flag:
                y_hat = self(batched_data).view(-1)
                y_gt = batched_data.y.view(-1).float()
                loss = self.loss_fn(y_hat, y_gt)
            else:
                y_gt = batched_data.y.view(-1).float()
                forward = lambda perturb: self(batched_data, perturb)
                model_forward = (self, forward)
                n_graph, n_node = batched_data.x.size()[:2]
                perturb_shape = (n_graph, n_node, self.hidden_dim)

                optimizer = self.optimizers()
                optimizer.zero_grad()
                loss, _ = flag_bounded(model_forward, perturb_shape, y_gt, optimizer, batched_data.x.device, self.loss_fn,
                               m=self.flag_m, step_size=self.flag_step_size, mag=self.flag_mag)
                self.lr_schedulers().step()
        else:
            y_hat = self(batched_data).view(-1)
            y_gt = batched_data.y.view(-1)
            loss = self.loss_fn(y_hat, y_gt)
        self.log('train_loss', loss, sync_dist=True)
        return loss

    def validation_step(self, batched_data, batch_idx):
        if self.dataset_name in ['PCQM4M-LSC', 'ZINC']:
            y_pred = self(batched_data).view(-1)
            y_true = batched_data.y.view(-1)
        else:
            y_pred = self(batched_data)
            y_true = batched_data.y
        return {
            'y_pred': y_pred,
            'y_true': y_true,
        }

    def validation_epoch_end(self, outputs):
        y_pred = torch.cat([i['y_pred'] for i in outputs])
        y_true = torch.cat([i['y_true'] for i in outputs])
        if self.dataset_name == 'ogbg-molpcba':
            mask = ~torch.isnan(y_true)
            loss = self.loss_fn(y_pred[mask], y_true[mask])
            self.log('valid_ap', loss, sync_dist=True)
        else:
            mask = y_true >= 1
            input_dict = {"y_true": y_true[mask], "y_pred": y_pred[mask]}
            try:
                self.log('valid_' + self.metric, self.evaluator.eval(input_dict)[self.metric], sync_dist=True)
                print(self.evaluator.eval(input_dict)[self.metric])
            except:
                pass

    def test_step(self, batched_data, batch_idx):
        if self.dataset_name in ['PCQM4M-LSC', 'ZINC']:
            y_pred = self(batched_data).view(-1)
            y_true = batched_data.y.view(-1)
        else:
            y_pred = self(batched_data)
            y_true = batched_data.y
        return {
            'y_pred': y_pred,
            'y_true': y_true,
            'idx': batched_data.idx,
        }

    def test_epoch_end(self, outputs):
        y_pred = torch.cat([i['y_pred'] for i in outputs])
        y_true = torch.cat([i['y_true'] for i in outputs])
        if self.dataset_name == 'PCQM4M-LSC':
            result = y_pred.cpu().float().numpy()
            idx = torch.cat([i['idx'] for i in outputs])
            torch.save(result, 'y_pred.pt')
            print(result.shape)
            exit(0)
        input_dict = {"y_true": y_true, "y_pred": y_pred}
        self.log('test_' + self.metric, self.evaluator.eval(input_dict)[self.metric], sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.peak_lr, weight_decay=self.weight_decay)
        lr_scheduler = {
            'scheduler': PolynomialDecayLR(
                optimizer,
                warmup_updates=self.warmup_updates,
                tot_updates=self.tot_updates,
                lr=self.peak_lr,
                end_lr=self.end_lr,
                power=1.0,
            ),
            'name': 'learning_rate',
            'interval':'step',
            'frequency': 1,
        }
        return [optimizer], [lr_scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("GraphFormer")
        parser.add_argument('--n_layers', type=int, default=12)
        parser.add_argument('--head_size', type=int, default=32)
        parser.add_argument('--hidden_dim', type=int, default=512)
        parser.add_argument('--ffn_dim', type=int, default=512)
        parser.add_argument('--intput_dropout_rate', type=float, default=0.1)
        parser.add_argument('--dropout_rate', type=float, default=0.1)
        parser.add_argument('--weight_decay', type=float, default=0.01)
        parser.add_argument('--attention_dropout_rate', type=float, default=0.1)
        parser.add_argument('--checkpoint_path', type=str, default='')
        parser.add_argument('--warmup_updates', type=int, default=60000)
        parser.add_argument('--tot_updates', type=int, default=1000000)
        parser.add_argument('--peak_lr', type=float, default=2e-4)
        parser.add_argument('--end_lr', type=float, default=1e-9)
        parser.add_argument('--edge_type', type=str, default='multi_hop')
        parser.add_argument('--validate', action='store_true', default=False)
        parser.add_argument('--test', action='store_true', default=False)
        parser.add_argument('--flag', action='store_true')
        parser.add_argument('--flag_m', type=int, default=3)
        parser.add_argument('--flag_step_size', type=float, default=1e-3)
        parser.add_argument('--flag_mag', type=float, default=1e-3)
        return parent_parser


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


tot_dis = [15, 7, 5, 3, 2, 1] # res_ao_2
# tot_dis = [15, 15, 7, 7, 5, 5, 3, 3, 2, 2, 1, 1] # res_ao_2
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, head_size, cur_layer):
        super(MultiHeadAttention, self).__init__()

        self.cur_layer = cur_layer
        self.head_size = head_size

        self.att_size = att_size = hidden_size // head_size
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, head_size * att_size)
        self.linear_k = nn.Linear(hidden_size, head_size * att_size)
        self.linear_v = nn.Linear(hidden_size, head_size * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(head_size * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None, rel_pos=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.head_size, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.head_size, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.head_size, d_v)

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        
        if attn_bias is not None:
            flags = []
            for i in range(self.head_size):
                cut_dist = tot_dis[self.cur_layer] + 0.5
                flag_new = (rel_pos <= cut_dist) + (attn_bias[:, i, 1:, 1:] < -1e8)
                flags.append(flag_new.unsqueeze(dim=1))
            flags = torch.cat(flags, dim=1)

            attn_bias[:, :, 1:, 1:] = attn_bias[:, :, 1:, 1:] * flags
            x = x + attn_bias

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.head_size * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, head_size, cur_layer):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, head_size, cur_layer)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

        self.rel_pos_encoder = nn.Embedding(512, head_size, padding_idx=0)
        # self.bcc_shortest_dis_encoder = nn.Embedding(512, head_size, padding_idx=0)

    def forward(self, x, attn_bias=None, rel_pos=None, bcc_shortest_dis=None):
        rel_pos_bias = self.rel_pos_encoder(rel_pos).permute(0, 3, 1, 2)
        # bcc_shortest_dis_bias = self.bcc_shortest_dis_encoder(bcc_shortest_dis).permute(0, 3, 1, 2)
        attn_bias[:, :, 1:, 1:] = attn_bias[:, :, 1:, 1:] + rel_pos_bias
        # attn_bias[:, :, 1:, 1:] = attn_bias[:, :, 1:, 1:] + rel_pos_bias + bcc_shortest_dis_bias
        
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias, rel_pos)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x
