# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import wraps

from models.BaseModel import SequentialModel
from utils import layers
import copy
import math
import random
import scipy.sparse as sp
from scipy.sparse import csr_matrix

class CL4rec(SequentialModel):
    reader = 'graphReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'beta', 'gamma', 'eta', 'tau', 'sim', 'lmd', 'c2']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=100,
                            help='Size of embedding vectors.')
        parser.add_argument('--beta', type=float, default=0.8,
                            help='Reorder')
        parser.add_argument('--gamma', type=float, default=0.5,
                            help='Mask')
        parser.add_argument('--eta', type=float, default=0.5,
                            help='Crop')
        parser.add_argument('--tau', type=int, default=1)
        parser.add_argument('--use_rl', type=int, default=1, help='I don t know why it is here, but it should be here')
        parser.add_argument('--sim', type=str, default='dot')
        parser.add_argument('--lmd', type=float, default=0.1, help='weight for L1')
        parser.add_argument('--c2', type=float, default=0.1, help='weight for L2')
        parser.add_argument('--num_gcn_layers', type=int, default=2, help='graph layer')
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self.max_his = args.history_max

        self.beta = args.beta
        self.eta = args.beta
        self.gamma = args.gamma
        self.mask_token = corpus.n_items
        self.tau = args.tau
        self.sim = args.sim
        self.lmd = args.lmd
        self.c2 = args.c2
        self.use_rl = args.use_rl
        # d, r, c = corpus.graph
        # iigraph = csr_matrix((d, (r, c)), shape=(corpus.n_items, corpus.n_items))
        # self.graph = self.make_torch_adj(iigraph).to_dense().cuda()
        # self.num_gcn_layers = args.num_gcn_layers
        # self.args = args
        self._define_params()
        self.optimizer_1 = None
        self.optimizer_2 = None
        self.apply(self.init_weights)

    def _define_params(self):
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        # self.fusion_layer_re = nn.Linear(self.emb_size * 2, self.emb_size)
        # self.fusion_layer = nn.Linear(self.emb_size * 2, self.emb_size)
        # self.graph_encoder = GraphEncoder(self.item_num, self.emb_size, self.num_gcn_layers)
        self.encoder = BERT4RecEncoder(self.emb_size, self.max_his, num_layers=2, num_heads=2)
        # self.g_encoder = BERT4RecEncoder(self.emb_size, self.max_his, num_layers=2, num_heads=2)
        # self.aug_1 = Extractor([self.emb_size, 32, 16, self.emb_size], "gelu", None)
        # self.aug_2 = Extractor([self.emb_size, 32, 16, self.emb_size], "gelu", None)
        # self.activation = F.tanh
        self.loss_fct = nn.CrossEntropyLoss()
        self.nce_fct = nn.CrossEntropyLoss()
        # self.model_parameter = [{"params": self.i_embeddings.parameters()}, {"params": self.graph_encoder.parameters()},
        #                         {"params": self.fusion_layer.parameters()},
        #                         {"params": self.fusion_layer_re.parameters()}, {"params": self.encoder.parameters()},
        #                         {"params": self.g_encoder.parameters()}]


    # def make_torch_adj(self, mat):
    #     mat = (mat + sp.eye(mat.shape[0]))
    #     mat = (mat != 0) * 1.0
    #     mat = self.normalize(mat)
    #     idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
    #     vals = torch.from_numpy(mat.data.astype(np.float32))
    #     shape = torch.Size(mat.shape)
    #     return torch.sparse.FloatTensor(idxs, vals, shape)

    # def normalize(self, mat):
    #     degree = np.array(mat.sum(axis=-1))
    #     dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
    #     dInvSqrt[np.isinf(dInvSqrt)] = 0.0
    #     dInvSqrtMat = sp.diags(dInvSqrt)
    #     return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

    def forward(self, feed_dict):
        self.check_list = []
        i_ids = feed_dict['item_id']  # bsz, n_candidate
        history = feed_dict['history_items']  # bsz, history_max
        lengths = feed_dict['lengths']  # bsz

        # adj, adj_his = self.graph_encoder(self.graph)
        i_vectors = self.i_embeddings(history)
        i_vector = self.encoder(i_vectors, lengths)
        i_emb = self.i_embeddings(i_ids)

        prediction = (i_vector[:, None, :] * i_emb).sum(-1)
        out_dict = {'out_seq': i_vector, 'prediction': prediction}

        # augmentation
        if feed_dict['phase'] == 'train':
            # test_item_emb = self.activation(self.fusion_layer(torch.cat((adj, self.i_embeddings.weight), 1)))
            logits = torch.matmul(i_vector, self.i_embeddings.weight.transpose(0, 1))
            loss = self.loss_fct(logits, i_ids[:, 0])

            out_dict['loss'] = loss
            aug1_history = feed_dict['history_items_a']  # bsz, history_max
            aug1_lengths = feed_dict['history_len_a']  # bsz
            aug2_history = feed_dict['history_items_a']  # bsz, history_max
            aug2_lengths = feed_dict['history_len_b']  # bsz
            aug1_his_vectors = self.i_embeddings(aug1_history)
            aug2_his_vectors = self.i_embeddings(aug2_history)
            aug1_his_vector = self.encoder(aug1_his_vectors, aug1_lengths)
            aug2_his_vector = self.encoder(aug2_his_vectors, aug2_lengths)
            out_dict['aug1_seq'] = aug1_his_vector
            out_dict['aug2_seq'] = aug2_his_vector
            # out_dict['aug3_seq'] = self.aug_1(aug1_his_vector)
            # out_dict['aug4_seq'] = self.aug_2(aug2_his_vector)

        return out_dict

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def info_nce(self, z_i, z_j, temp, batch_size, sim='dot'):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)
        if sim == 'cos':
            sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        elif sim == 'dot':
            sim = torch.mm(z, z.T) / temp
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(batch_size)
        negative_samples = sim[mask].reshape(N, -1)
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels

    def contrast(self, z_i, z_j, sim='dot'):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        batch_size = z_i.shape[0]
        temp = self.tau
        N = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)
        if sim == 'cos':
            sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        elif sim == 'dot':
            sim = torch.mm(z, z.T) / temp
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(batch_size)
        negative_samples = sim[mask].reshape(N, -1)
        return positive_samples, negative_samples

    def meta_contrast_rl(self, a, b):
        ori_p, ori_n = self.contrast(a, b, "dot")
        min_positive_value, min_pos_pos = torch.min(ori_p, dim=-1)
        max_negative_value, max_neg_pos = torch.max(ori_n, dim=-1)
        lgamma_margin_pos, _ = torch.min(torch.cat((min_positive_value.unsqueeze(1), max_negative_value
                                                    .unsqueeze(1)), dim=1), dim=-1)
        lgamma_margin_pos = lgamma_margin_pos.unsqueeze(1)
        lgamma_margin_neg, _ = torch.max(torch.cat((min_positive_value.unsqueeze(1), max_negative_value
                                                    .unsqueeze(1)), dim=1), dim=-1)
        lgamma_margin_neg = lgamma_margin_neg.unsqueeze(1)
        loss = torch.mean(torch.clamp(ori_p - lgamma_margin_pos, min=0))
        loss += torch.mean(torch.clamp(lgamma_margin_neg - ori_n, min=0))
        return loss

    def meta_contrast(self, sequence_output_0, sequence_output_1, sequence_output_2, sequence_output_3, mode):
        """
        :param sequence_output_0:original seq1
        :param sequence_output_1: original seq2
        :param meta_aug: [aug_1,aug_2]
        :param mode: "step1 ,2, 3"
        :param weights: "aug weight"
        :return:
        """
        batch_size = sequence_output_0.shape[0]
        use_rl = self.use_rl
        # -------------------------------------------------step1-------------------------------------------------
        if mode == "step1":

            logits, label = self.info_nce(sequence_output_0, sequence_output_3, 1.0, batch_size, "dot")
            cl_loss_0 = nn.CrossEntropyLoss()(logits, label)

            logits, label = self.info_nce(sequence_output_1, sequence_output_2, 1.0, batch_size, "dot")
            cl_loss_1 = nn.CrossEntropyLoss()(logits, label)

            logits, label = self.info_nce(sequence_output_2, sequence_output_3, 1.0, batch_size, "dot")
            cl_loss_2 = nn.CrossEntropyLoss()(logits, label)
            cl_loss = cl_loss_0 + cl_loss_1 + cl_loss_2
            if use_rl:
                cl_loss += self.meta_rl(sequence_output_0, sequence_output_1, sequence_output_2, sequence_output_3)
        # -------------------------------------------------step2-------------------------------------------------
        elif mode == "step2":
            logits, label = self.info_nce(sequence_output_0, sequence_output_3, 1.0, batch_size, "dot")
            cl_loss_0 = nn.CrossEntropyLoss()(logits, label)

            logits, label = self.info_nce(sequence_output_1, sequence_output_2, 1.0, batch_size, "dot")
            cl_loss_1 = nn.CrossEntropyLoss()(logits, label)

            logits, label = self.info_nce(sequence_output_2, sequence_output_3, 1.0, batch_size, "dot")
            cl_loss_2 = nn.CrossEntropyLoss()(logits, label)
            cl_loss = cl_loss_0 + cl_loss_1 + cl_loss_2
            if use_rl:
                cl_loss += self.meta_rl(sequence_output_0, sequence_output_1, sequence_output_2, sequence_output_3)
        # -------------------------------------------------step3-------------------------------------------------
        else:
            logits, label = self.info_nce(sequence_output_0, sequence_output_3, 1.0, batch_size, "dot")
            cl_loss_0 = nn.CrossEntropyLoss()(logits, label)
            logits, label = self.info_nce(sequence_output_1, sequence_output_2, 1.0, batch_size, "dot")
            cl_loss_1 = nn.CrossEntropyLoss()(logits, label)
            logits, label = self.info_nce(sequence_output_2, sequence_output_3, 1.0, batch_size, "dot")
            cl_loss_2 = nn.CrossEntropyLoss()(logits, label)
            cl_loss = cl_loss_0 + cl_loss_1 + cl_loss_2
            if use_rl:
                cl_loss += self.meta_rl(sequence_output_0, sequence_output_1, sequence_output_2, sequence_output_3)
        return cl_loss

    def meta_rl(self, sequence_output_0, sequence_output_1, sequence_output_2, sequence_output_3):
        rl_loss = 0.
        rl_loss += self.meta_contrast_rl(sequence_output_0, sequence_output_3)
        rl_loss += self.meta_contrast_rl(sequence_output_1, sequence_output_2)
        rl_loss += self.meta_contrast_rl(sequence_output_2, sequence_output_3)
        return 0.1 * rl_loss

    def loss(self, out_dict: dict, mode) -> torch.Tensor:
        loss = out_dict['loss']
        seq_output1, seq_output2 = out_dict['aug1_seq'], out_dict['aug2_seq']
        seq_output3, seq_output4 = out_dict['aug3_seq'], out_dict['aug4_seq']
        nce_logits, nce_labels = self.info_nce(seq_output1, seq_output2, temp=self.tau, batch_size=seq_output1.shape[0],
                                               sim=self.sim)
        nce_loss = self.nce_fct(nce_logits, nce_labels)

        loss += self.lmd * nce_loss
        # if mode == "step1":
        #     loss += self.beta * self.meta_contrast(seq_output1, seq_output2, seq_output3, seq_output4, mode)
        #     return loss
        # elif mode == "step2":
        #     loss = 0.
        #     loss += self.meta_contrast(seq_output1, seq_output2, seq_output3, seq_output4, mode)
        #     return loss
        # else:
        #     loss += self.beta * self.meta_contrast(seq_output1, seq_output2, seq_output3, seq_output4, mode)
        #     return loss

        return loss

    class Dataset(SequentialModel.Dataset):
        #    class Dataset(SequentialModel.Dataset):
        def item_crop(self, seq):
            num_left = math.floor(len(seq) * self.model.eta)
            crop_begin = random.randint(0, len(seq) - num_left)
            croped_item_seq = np.zeros(len(seq), dtype= 'int')
            if crop_begin + num_left < seq.shape[0]:
                croped_item_seq[:num_left] = seq[crop_begin:crop_begin + num_left]
            else:
                croped_item_seq[:num_left] = seq[crop_begin:]
            return croped_item_seq, num_left

        def item_mask(self, seq):
            num_mask = math.floor(len(seq) * self.model.gamma)
            mask_index = random.sample(range(len(seq)), k=num_mask)
            masked_item_seq = seq#.tolist()
            # masked_item_seq = torch.tensor(masked_item_seq, dtype=torch.long)
            masked_item_seq[mask_index] = 0
            return masked_item_seq, len(seq)

        def item_reorder(self, seq):
            num_reorder = math.floor(len(seq) * self.model.beta)
            reorder_begin = random.randint(0, len(seq) - num_reorder)
            reordered_item_seq = seq#.tolist()
            subsequence = reordered_item_seq[reorder_begin:reorder_begin + num_reorder]
            random.shuffle(subsequence)
            reordered_item_seq = np.concatenate((reordered_item_seq[:reorder_begin], subsequence, reordered_item_seq[
                                                                                    reorder_begin + num_reorder:]))
            # reordered_item_seq = np.array(reordered_item_seq)
            # reordered_item_seq = torch.tensor(reordered_item_seq, dtype=torch.long)
            return reordered_item_seq, len(seq)

        def augment(self, seq, switch):
            switch = [switch]
            if switch[0] == 0:
                aug_seq, aug_len = self.item_crop(seq)
            elif switch[0] == 1:
                aug_seq, aug_len = self.item_mask(seq)
            elif switch[0] == 2:
                aug_seq, aug_len = self.item_reorder(seq)
            return aug_seq, aug_len

        def _get_feed_dict(self, index):
            feed_dict = super()._get_feed_dict(index)
            if self.phase == 'train':
                if feed_dict['lengths'] == 1:
                    switch = [3, 3]
                    feed_dict['history_items_a'] = feed_dict['history_items']
                    feed_dict['history_items_b'] = feed_dict['history_items']
                    feed_dict['history_len_a'] = feed_dict['lengths']
                    feed_dict['history_len_b'] = feed_dict['lengths']
                else:
                    switch = random.sample(range(3), k=2)
                    history_items_a, history_len_a = self.augment(feed_dict['history_items'], switch[0])
                    history_items_b, history_len_b = self.augment(feed_dict['history_items'], switch[1])
                    feed_dict['history_items_a'] = history_items_a
                    feed_dict['history_items_b'] = history_items_b
                    feed_dict['history_len_a'] = history_len_a
                    feed_dict['history_len_b'] = history_len_b
            return feed_dict


""" Encoder Layer """


class BERT4RecEncoder(nn.Module):
    def __init__(self, emb_size, max_his, num_layers=2, num_heads=2):
        super().__init__()
        self.p_embeddings = nn.Embedding(max_his + 1, emb_size)
        self.transformer_block = nn.ModuleList([
            layers.TransformerLayer(d_model=emb_size, d_ff=emb_size, n_heads=num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, seq, lengths):
        batch_size, seq_len = seq.size(0), seq.size(1)
        len_range = torch.from_numpy(np.arange(seq_len)).to(seq.device)
        valid_mask = len_range[None, :] < lengths[:, None]

        # Position embedding
        position = len_range[None, :] * valid_mask.long()
        pos_vectors = self.p_embeddings(position)
        seq = seq + pos_vectors

        # Self-attention
        attn_mask = valid_mask.view(batch_size, 1, 1, seq_len)
        for block in self.transformer_block:
            seq = block(seq, attn_mask)
        seq = seq * valid_mask[:, :, None].float()

        his_vector = seq[torch.arange(batch_size), lengths - 1]
        return his_vector


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different
        (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) *
        (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish}


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


# Augmenter
class Extractor(nn.Module):
    def __init__(self, layers, activation='gelu', init_method=None):
        super(Extractor, self).__init__()
        self.dense_1 = nn.Linear(layers[0], layers[1])
        self.dense_2 = nn.Linear(layers[1], layers[2])
        self.dense_3 = nn.Linear(layers[2], layers[3])
        if activation != None:
            self.actfunction = ACT2FN[activation]
        self.normal = Normalize()

    def forward(self, input):  # [B H]
        # layer 1
        output = self.dense_1(input)
        if self.actfunction != None:
            output = self.actfunction(output)
        # layer 2
        output = self.dense_2(output)
        if self.actfunction != None:
            output = self.actfunction(output)
        # layer 3
        output = self.dense_3(output)
        output = self.normal(output)
        return output

class GraphEncoder(nn.Module):
    def __init__(self, item_num, emb_size, num_gcn_layers):
        super(GraphEncoder, self).__init__()
        init = nn.init.xavier_uniform_
        self.item_emb = nn.Embedding(item_num, emb_size)  # args.item = num_real_item + 1
        self.gcn_layers = nn.Sequential(*[GCNLayer() for i in range(num_gcn_layers)])

    def get_ego_embeds(self):
        return self.item_emb.weight

    def forward(self, encoder_adj):
        embeds = [self.item_emb.weight]
        for i, gcn in enumerate(self.gcn_layers):
            embeds.append(gcn(encoder_adj, embeds[-1]))
        return sum(embeds), embeds

class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()

    def forward(self, adj, embeds):
        return torch.spmm(adj, embeds)

