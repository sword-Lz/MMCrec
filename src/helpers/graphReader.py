# -*- coding: UTF-8 -*-

import logging
import pandas as pd
import copy
import numpy as np
from scipy.sparse import csr_matrix
import torch
from helpers.BaseReader import BaseReader
import scipy.sparse as sp

class graphReader(BaseReader):
    def __init__(self, args):
        super().__init__(args)
        self._append_his_info()
        self._construct_graph(3)
    def _append_his_info(self):
        """
        self.user_his: store user history sequence [(i1,t1), (i1,t2), ...]
        add the 'position' of each interaction in user_his to data_df
        """
        logging.info('Appending history info...')
        sort_df = self.all_df.sort_values(by=['time', 'user_id'], kind='mergesort')
        position = list()
        self.user_his = dict()  # store the already seen sequence of each user
        for uid, iid, t in zip(sort_df['user_id'], sort_df['item_id'], sort_df['time']):
            if uid not in self.user_his:
                self.user_his[uid] = list()
            position.append(len(self.user_his[uid]))
            self.user_his[uid].append((iid, t))
        sort_df['position'] = position
        for key in ['train', 'dev', 'test']:
            self.data_df[key] = pd.merge(
                left=self.data_df[key], right=sort_df, how='left',
                on=['user_id', 'item_id', 'time'])
        del sort_df

    def _construct_graph(self, distance):
        train_data = self.data_df['train']
        df_sorted = train_data.sort_values(by=['user_id', 'time'])
        num_items = self.n_items
        seqs = []
        import tqdm
        for user_id in tqdm.tqdm(df_sorted['user_id'].unique()):
            seqs.append(df_sorted[df_sorted['user_id'] == user_id]['item_id'].tolist())
        r, c, d = list(), list(), list()
        for i, seq in enumerate(seqs):
            print(f"Processing {i}/{len(seqs)}          ", end='\r')
            for dist in range(1, distance + 1):
                if dist >= len(seq): break;
                r += copy.deepcopy(seq[+dist:])
                c += copy.deepcopy(seq[:-dist])
                r += copy.deepcopy(seq[:-dist])
                c += copy.deepcopy(seq[+dist:])
        d = np.ones_like(r)
        self.graph = (d, r, c)
        # iigraph = csr_matrix((d, (r, c)), shape=(num_items, num_items))
        # self.graph = self.make_torch_adj(iigraph)
        torch.save(self.graph, 'graph.pt')

        # 从文件中加载张量

        print('Constructed i-i graph, density=%.6f' % (len(d) / (num_items ** 2)))
    def make_torch_adj(self, mat):
        mat = (mat + sp.eye(mat.shape[0]))
        mat = (mat != 0) * 1.0
        mat = self.normalize(mat)
        idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = torch.from_numpy(mat.data.astype(np.float32))
        shape = torch.Size(mat.shape)
        return torch.sparse.FloatTensor(idxs, vals, shape)
    def normalize(self, mat):
        degree = np.array(mat.sum(axis=-1))
        dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
        dInvSqrt[np.isinf(dInvSqrt)] = 0.0
        dInvSqrtMat = sp.diags(dInvSqrt)
        return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()