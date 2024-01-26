import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dgl.nn.pytorch.softmax import edge_softmax
from utils.helper import edge_softmax_fix
from modules.hyperbolic import *


def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


class Aggregator(nn.Module):

    def __init__(self, in_dim, out_dim, dropout, aggregator_type, n_params):
        super(Aggregator, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.aggregator_type = aggregator_type

        self.n_users = n_params['n_users']
        self.n_entities = n_params['n_entities']
        self.n_relations = n_params['n_relations']
        self.n_items = n_params['n_items']

        self.message_dropout = nn.Dropout(dropout)

        self.gate1 = nn.Linear(self.in_dim, self.out_dim, bias=False)
        self.gate2 = nn.Linear(self.in_dim, self.out_dim, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, mode, sg, g_i2u, g_u2i, entity_embed, user_embed, relation_emb, sg_inv):
        
        sg = sg.local_var()
        # o = torch.zeros_like(entity_embed[:, 0].view(-1, 1))
        sg.ndata['node'] = entity_embed
        sg.ndata['node1'] = expmap0(entity_embed)

        def tan_sum(edges):
            tan_sum = logmap(project(mobius_add(expmap(edges.dst['node'], expmap0(edges.src['node'])), expmap(relation_emb[edges.data['type'] + 2], expmap0(edges.src['node'])))), expmap0(edges.src['node']))
            # tan_sum = logmap0(expmap(ptransp0(edges.src['node1'], edges.dst['node']) + ptransp0(edges.src['node1'], relation_emb[edges.data['type'] + 2]), edges.src['node1']))
            return {'tan_sum': tan_sum}

        sg.apply_edges(tan_sum, sg.edges(form='all')[2])

        sg_inv.edata['tan_sum'] = sg.edata['tan_sum']

        sg_inv.update_all(dgl.function.copy_e('tan_sum', 'temp'), dgl.function.mean('temp', 'out'))

        out = sg_inv.ndata['out']
        out = self.message_dropout(out)

       
        g_i2u = g_i2u.local_var()
        g_u2i = g_u2i.local_var()

        norm_all = (torch.norm(torch.cat([out[:self.n_items], user_embed], dim=0), dim=1).view(-1, 1)) ** 2

        g_i2u.ndata['node'] = torch.cat([out[:self.n_items], user_embed], dim=0)
        g_i2u.update_all(dgl.function.copy_u('node', 't'), dgl.function.mean('t', 'u'))
        u = g_i2u.ndata['u'][self.n_items:]

        g_u2i.ndata['node'] = torch.cat([out[:self.n_items], user_embed], dim=0) / (norm_all + 1e-6)
        g_u2i.ndata['norm'] = norm_all

        g_u2i.update_all(dgl.function.u_mul_v('node', 'norm', 't'), dgl.function.mean('t', 'u'))

        # g_u2i.ndata['node'] = torch.cat([out[:self.n_items], user_embed], dim=0)
        # g_u2i.update_all(dgl.function.copy_u('node','t'), dgl.function.mean('t', 'u'))

        i_cf = g_u2i.ndata['u'][:self.n_items]

        gi = self.sigmoid(self.gate1(out[:self.n_items]) + self.gate2(i_cf))
        item_emb_fusion = (gi * out[:self.n_items]) + ((1 - gi) * i_cf)

        # item_emb_fusion = torch.norm(out[:self.n_items], dim=1).view(-1, 1) * F.normalize(item_emb_fusion)

        return torch.cat([item_emb_fusion, out[self.n_items:]], dim=0), u, out[:self.n_items]


class DHN(nn.Module):

    def __init__(self, args, n_params,
                 user_pre_embed=None, item_pre_embed=None):

        super(DHN, self).__init__()
        self.use_pretrain = args.use_pretrain

        self.n_users = n_params['n_users']
        self.n_entities = n_params['n_entities']
        self.n_relations = n_params['n_relations']
        self.n_items = n_params['n_items']

        self.entity_dim = args.entity_dim
        self.relation_dim = args.relation_dim

        self.aggregation_type = args.aggregation_type
        self.conv_dim_list = [args.entity_dim] + eval(args.conv_dim_list)  
        self.mess_dropout = eval(args.mess_dropout)  # layers dropout
        self.n_layers = len(eval(args.conv_dim_list))  
        self.dropout = nn.Dropout(p=0.1)

        self.kg_l2loss_lambda = args.kg_l2loss_lambda
        self.cf_l2loss_lambda = args.cf_l2loss_lambda

        self.num_neg_sample = args.num_neg_sample
        self.margin_ccl = args.margin
        self.device = torch.device("cuda:" + str(args.gpu_id))

        # Embedding
        self.relation_embed = nn.Embedding(self.n_relations, self.relation_dim)
        self.entity_user_embed = nn.Embedding(self.n_entities + self.n_users, self.entity_dim)
        if (self.use_pretrain == 1) and (user_pre_embed is not None) and (item_pre_embed is not None):
            other_entity_embed = nn.Parameter(torch.Tensor(self.n_entities - item_pre_embed.shape[0], self.entity_dim))
            nn.init.xavier_uniform_(other_entity_embed, gain=nn.init.calculate_gain('relu'))  # 均匀分布
            entity_user_embed = torch.cat([item_pre_embed, other_entity_embed, user_pre_embed], dim=0)
            self.entity_user_embed.weight = nn.Parameter(entity_user_embed)

        self.items_embed_cf = nn.Embedding(self.n_items, self.entity_dim)

        self.aggregator_layers = nn.ModuleList()
        for k in range(self.n_layers):
            self.aggregator_layers.append(
                Aggregator(self.conv_dim_list[k], self.conv_dim_list[k + 1], self.mess_dropout[k],
                           self.aggregation_type, n_params))

    def cf_embedding(self, mode, g_kg, g_i2u, g_u2i):
        g_i2u = g_i2u.local_var()
        g_u2i = g_u2i.local_var()
        g_kg = g_kg.local_var()

        g_kg_inv = dgl.graph((g_kg.edges()[1], g_kg.edges()[0]))

        idx = np.random.choice(g_kg.all_edges(form='all')[2].shape[0], size=int(g_kg.all_edges(form='all')[2].shape[0] * 0.5), replace=False)
        sg = dgl.edge_subgraph(g_kg, idx, relabel_nodes=False)
        sg_inv = dgl.edge_subgraph(g_kg_inv, idx, relabel_nodes=False)

        ego_embed = self.entity_user_embed(sg.ndata['id'])

        u_embed = self.entity_user_embed.weight[self.n_entities:]

        entities_res = ego_embed
        out_res = ego_embed[:self.n_items]
        user_res = u_embed

        for i, layer in enumerate(self.aggregator_layers):
            ego_embed, u_embed, out = layer(mode, sg, g_i2u, g_u2i, ego_embed, u_embed, self.relation_embed.weight, sg_inv)
            # message dropout
            ego_embed = self.dropout(ego_embed)
            u_embed = self.dropout(u_embed)
            out = self.dropout(out)

            ego_embed = F.normalize(ego_embed)
            u_embed = F.normalize(u_embed)
            out = F.normalize(out)

            entities_res = torch.add(entities_res, ego_embed)
            user_res = torch.add(user_res, u_embed)
            out_res = torch.add(out_res, out)

        loss = self.graph_kg_loss(sg, torch.cat([out_res, entities_res[self.n_items:]], dim=0))

        return entities_res, user_res, loss

    def cf_score(self, mode, g_kg, g_i2u, g_u2i):
        """
        user_ids:   number of users to evaluate   (n_eval_users)
        item_ids:   number of items to evaluate   (n_eval_items)
        """
        entities_embed, users_embed, _ = self.cf_embedding(mode, g_kg, g_i2u, g_u2i)  # (n_users + n_entities, cf_concat_dim)
        return users_embed, entities_embed[:self.n_items]

    def all_pos_loss(self, g, u, i, n_i, u_id):
        g = g.local_var()

        g.ndata['node'] = torch.cat([i, u], dim=0)

        g.ndata['node1'] = lor_expmap0(torch.cat([torch.zeros_like(torch.cat([i, u], dim=0)[:, 0]).view(-1, 1), torch.cat([i, u], dim=0)], dim=1))
        sg = dgl.in_subgraph(g, u_id + n_i)

        def i_p(edges):
            i_per = 1 - torch.cosine_similarity(edges.dst['node'], edges.src['node'], dim=1)
            l_pos = lor_distance(edges.dst['node1'], edges.src['node1'])
            return {'i_p': i_per, 'l_p': l_pos}

        sg.apply_edges(i_p, sg.edges(form='all')[2])

        sg.update_all(dgl.function.copy_e('l_p', 't'), dgl.function.mean('t', 'loss1'))
        sg.update_all(dgl.function.copy_e('i_p', 't'), dgl.function.mean('t', 'loss'))
        pos = sg.ndata['loss'][u_id + n_i]
        pos1 = sg.ndata['loss1'][u_id + n_i]
        return pos, pos1

    def calc_cf_loss(self, mode, g_kg, g_i2u, g_u2i, user_ids, item_pos_ids, item_neg_ids, epoch):
        """
        user_ids:       (cf_batch_size)
        item_pos_ids:   (cf_batch_size)
        item_neg_ids:   (cf_batch_size, N_num_neg)
        """
        entities_embed, users_embed, kg_loss = self.cf_embedding(mode, g_kg, g_i2u, g_u2i)  # (n_users + n_entities, cf_concat_dim)
        
        item_neg_ids = item_neg_ids.view(-1)

        u_e = users_embed[user_ids]

        pos_e, neg_e = entities_embed[item_pos_ids], entities_embed[item_neg_ids]

        neg_score, neg_dist = self.create_contrastive_loss(u_e, neg_e)
        pos_score, pos_dist = self.all_pos_loss(g_i2u, users_embed, entities_embed[:self.n_items], self.n_items, user_ids)

        cf_loss_ang = torch.mean(pos_score + neg_score)
        cf_loss_dist = torch.mean(pos_dist - neg_dist)
        
        return cf_loss_ang - 1e-4 * _L2_loss_mean(entities_embed[:self.n_items]) + 1e-2*cf_loss_dist + 1e-1 * kg_loss


    def graph_kg_loss(self, g_kg, embed):
        g_kg = g_kg.local_var()
       
        g_kg_inv = dgl.graph((g_kg.edges()[1], g_kg.edges()[0]))

        
        # idx = np.random.choice(g_kg.all_edges(form='all')[2].shape[0], size=max(int(g_kg.all_edges(form='all')[2].shape[0] * 0.5 * (0.995 ** epoch)), 2048), replace=False)
        # sg = dgl.edge_subgraph(g_kg, idx, relabel_nodes=False)
        # ssg = dgl.node_subgraph(sg, torch.arange(self.n_items, self.n_entities).to(self.device))
        # sg_inv = dgl.edge_subgraph(g_kg_inv, idx, relabel_nodes=False)
        # ssg_inv = dgl.node_subgraph(sg_inv, torch.arange(self.n_items, self.n_entities).to(self.device))

        sg = g_kg
        sg_inv = g_kg_inv
        relation_emb = self.relation_embed.weight

        sg.ndata['node'] = embed
        sg.ndata['node1'] = expmap0(embed)

        def kg_loss(edges):
            # sub = hyp_distance(expmap(ptransp0(edges.src['node1'], edges.dst['node']) + ptransp0(edges.src['node1'], relation_emb[edges.data['type'] + 2]), edges.src['node1']), edges.src['node1'])
            sub = hyp_distance(expmap0(logmap(project(mobius_add(expmap(edges.dst['node'], expmap0(edges.src['node'])), expmap(self.relation_embed(edges.data['type'] + 2), expmap0(edges.src['node'])))), expmap0(edges.src['node']))), expmap0(edges.src['node']))
            return {'sub': sub}

        sg.apply_edges(kg_loss, sg.edges(form='all')[2])
        sg_inv.edata['sub'] = sg.edata['sub']
        sg_inv.update_all(dgl.function.copy_e('sub', 'temp'), dgl.function.mean('temp', 'out'))
        loss = torch.sum(sg_inv.ndata['out'])/(torch.sum(sg_inv.ndata['out']>0))

        return loss

    def rate1(self, u_g_embeddings, i_g_embeddings):
        u = torch.unsqueeze(lor_expmap0(torch.cat([torch.zeros_like(u_g_embeddings[:, 0]).view(-1, 1), u_g_embeddings], dim=1)), dim=1)
        i = lor_expmap0(torch.cat([torch.zeros_like(i_g_embeddings[:, 0]).view(-1, 1), i_g_embeddings], dim=1))[None, ...]
        dis = lor_distance(u,i).view(u.shape[0],i.shape[1])
        ang = torch.cosine_similarity(u_g_embeddings.unsqueeze(1), i_g_embeddings.unsqueeze(0), dim=2)
        a = 0.5
        s = a * torch.sigmoid(1/dis) + (1-a) * ang
        return s.detach().cpu()

    def rate(self, u_g_embeddings, i_g_embeddings):
        return torch.cosine_similarity(u_g_embeddings.unsqueeze(1), i_g_embeddings.unsqueeze(0), dim=2).detach().cpu()

    def forward(self, mode, *input):
        if mode == 'calc_att':
            return self.compute_attention(*input)
        if mode == 'calc_cf_loss':
            return self.calc_cf_loss(mode, *input)
        if mode == 'calc_kg_loss':
            return self.calc_kg_loss(*input)
        if mode == 'graph_kg_loss':
            return self.graph_kg_loss(*input)
        if mode == 'predict':
            return self.cf_score(mode, *input)
        if mode == 'rating':
            return self.rate(*input)

    def create_contrastive_loss(self, u_e, neg_e):
        batch_size = u_e.shape[0]

        u_e_lor = lor_expmap0(torch.cat([torch.zeros_like(u_e[:, 0]).view(-1, 1), u_e], dim=1))
        neg_e_lor = lor_expmap0(torch.cat([torch.zeros_like(neg_e[:, 0]).view(-1, 1), neg_e], dim=1))

        u_e = F.normalize(u_e)
        neg_e = F.normalize(neg_e)

        users_batch = torch.repeat_interleave(u_e, self.num_neg_sample, dim=0)

        ui_neg1 = torch.relu(torch.cosine_similarity(users_batch, neg_e, dim=1) - self.margin_ccl)
        ui_neg1 = ui_neg1.view(batch_size, -1)
        x = ui_neg1 > 0
        ui_neg_loss1 = torch.sum(ui_neg1, dim=-1) / (torch.sum(x, dim=-1) + 1e-5)

        users_batch_lor = torch.repeat_interleave(u_e_lor, self.num_neg_sample, dim=0)

        ui_neg2 = lor_distance(users_batch_lor, neg_e_lor).view(batch_size, -1)
        ui_neg_loss2 = torch.mean(ui_neg2, dim=-1)

        return ui_neg_loss1, ui_neg_loss2
