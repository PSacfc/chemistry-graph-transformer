# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import numpy as np
import torch_geometric.datasets
from pcq_wrapper import MyPygPCQM4MDataset
from ogb.lsc.pcqm4m_pyg import PygPCQM4MDataset
import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})
import algos
import joblib
import networkx as nx

def convert_to_single_emb(x, offset=128):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x

def cal_bcc(data):
    N = data.num_nodes
    shortest_dis = np.zeros((N, N))
    graph = nx.Graph()
    belong = np.zeros(data.num_nodes)
    belong[:] = -1
    for i in range(data.edge_index.shape[1]):
        u = data.edge_index[0, i]
        v = data.edge_index[1, i]
        if u < v:
            graph.add_edge(u.item(), v.item())
    cnt = 0
    for i in nx.biconnected_components(graph):
        if len(i) > 2:
            for node in i:
                belong[node] = cnt
            cnt += 1
            
    for i in range(data.num_nodes):
        if belong[i] == -1:
            belong[i] = cnt
            cnt += 1
    new_graph = nx.Graph()
    for i in range(data.edge_index.shape[1]):
        u = belong[data.edge_index[0, i]]
        v = belong[data.edge_index[1, i]]
        if u < v:
            new_graph.add_edge(u.item(), v.item())
            
    new_shortest_dis = np.zeros((cnt, cnt))
    path = nx.all_pairs_dijkstra_path_length(new_graph)
    
    for i in path:
        s, p = int(i[0]), i[1]
        for j in range(cnt):
            try:
                new_shortest_dis[s, j] = p[j]
            except:
                new_shortest_dis[s, j] = -1

    for i in range(N):
        for j in range(N):
            shortest_dis[i, j] = new_shortest_dis[int(belong[i]), int(belong[j])]
            if shortest_dis[i, j] > 510:
                shortest_dis[i, j] = 510

    bcc_shortest_dis = np.zeros((N, N))
    for j, bcc in enumerate(nx.biconnected_components(graph)):
        if len(bcc) <= 2:
            continue
        bcc_in_degree = np.zeros(N)
        temp_g = nx.Graph()
        temp_node = 0
        for tt in bcc:
            temp_node = tt
        for s, t in zip(data.edge_index[0], data.edge_index[1]):
            if belong[s] == belong[t] and belong[s] == belong[temp_node] and s < t:
                temp_g.add_edge(s.item(), t.item())
                bcc_in_degree[s] = bcc_in_degree[s] + 1
                bcc_in_degree[t] = bcc_in_degree[t] + 1
        
        paths = nx.all_pairs_dijkstra_path(temp_g)
        for path in paths:
            st, ways = path
            for way in ways:
                if len(ways[way]) == 1:
                    continue
                curr_path = ways[way]
                ed = curr_path[-1]
                for idx, now in enumerate(curr_path):
                    if idx > 0:
                        if bcc_in_degree[now] >= 3 and bcc_in_degree[curr_path[idx - 1]] < 3:
                                bcc_shortest_dis[st, ed] = bcc_shortest_dis[st, ed] + 1

    return shortest_dis, belong, bcc_shortest_dis

def preprocess_item(item, noise=False):
    edge_attr, edge_index, x = item.edge_attr, item.edge_index, item.x
    N = x.size(0)
    x = convert_to_single_emb(x)

    # node adj matrix [N, N] bool
    adj = torch.zeros([N, N], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True

    # edge feature here
    if len(edge_attr.size()) == 1:
        edge_attr = edge_attr[:, None]
    
    all_rel_pos_3d_with_noise = torch.from_numpy(algos.bin_rel_pos_3d_1(item.all_rel_pos_3d, noise=noise)).long()
    rel_pos_3d_attr = all_rel_pos_3d_with_noise[edge_index[0, :], edge_index[1, :]]
    edge_attr = torch.cat([edge_attr, rel_pos_3d_attr[:, None]], dim=-1)
    attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    attn_edge_type[edge_index[0, :], edge_index[1, :]] = convert_to_single_emb(edge_attr) + 1

    shortest_path_result, path = algos.floyd_warshall(adj.numpy())
    max_dist = np.amax(shortest_path_result)
    edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
    # rel_pos = torch.from_numpy((shortest_path_result)).long()
    attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float) # with graph token
    
    new_shortest_dis, belong, bcc_shortest_dis = cal_bcc(item)
    for i in range(N):
        for j in range(N):
            if belong[i] != belong[j]:
                shortest_path_result[i, j] = -1
    shortest_path_result = shortest_path_result + 1
    bcc_shortest_dis = bcc_shortest_dis + 1
    inner_rel_pos = torch.from_numpy((shortest_path_result)).long()
    rel_pos = torch.from_numpy((new_shortest_dis)).long()

    # combine
    item.x = x
    item.adj = adj
    item.attn_bias = attn_bias
    item.attn_edge_type = attn_edge_type
    item.rel_pos = rel_pos
    item.in_degree = adj.long().sum(dim=1).view(-1)
    item.out_degree = adj.long().sum(dim=0).view(-1)
    item.edge_input = torch.from_numpy(edge_input).long()
    item.bcc_shortest_dis = torch.from_numpy(bcc_shortest_dis).long()

    item.all_rel_pos_3d_1 = torch.from_numpy(item.all_rel_pos_3d).float()
    return item

class MyPygPCQM4MDataset2(MyPygPCQM4MDataset):
    def __init__(self, root = 'dataset/mypcq_v4'):
        super().__init__(root=root)
        self.all_rel_pos_3d = joblib.load('dataset/all_rel_pos_3d.pkl')

    def download(self):
        super(MyPygPCQM4MDataset2, self).download()

    def process(self):
        super(MyPygPCQM4MDataset2, self).process()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx])
            item.idx = idx
            item.all_rel_pos_3d = self.all_rel_pos_3d[self.indices()[idx]]
            # donot add noise to test molecules
            if self.indices()[idx] >= 3426030:
                return preprocess_item(item, noise=False)
            return preprocess_item(item, noise=True)
        else:
            return self.index_select(idx)