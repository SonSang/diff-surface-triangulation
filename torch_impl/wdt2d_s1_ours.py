import torch as th
from torch_impl.wdt2d_ours import WDT as WDT_OURS_S2
from torch_impl.wdt2d_ours import tensor_intersect, tensor_subtract, callback0

def callback0(edge_hp: th.Tensor,
                edge_cc: th.Tensor,
                hp_normal: th.Tensor,
                hp_height: th.Tensor):

    '''
    @edge_hp: [# pair, (2 + 1) + (2 + 1)]
    '''
    
    edges = edge_hp[:, :3]
    hplanes = edge_hp[:, 3:]
    
    assert th.all(edges[:, 0] == hplanes[:, 0]), ""
    
    edge_id = edges[:, -1]
    hp_id = hplanes[:, -1]
    
    curr_edge_cc = edge_cc[edge_id]
    curr_hp_normal = hp_normal[hp_id]
    curr_hp_height = hp_height[hp_id]
    
    # compute distance
    curr_dist = -th.sum(curr_hp_normal * curr_edge_cc, dim=-1, keepdim=True) + curr_hp_height
    # curr_dist = th.abs(curr_dist)
    
    # filter
    invalid = edges[:, 1] == hplanes[:, 1]
    curr_dist[invalid] = float('inf')
    
    result = th.cat([edges, hplanes, curr_dist], dim=-1)
    
    return result

def callback1(edge_cc: th.Tensor,
            edge_hp_normal: th.Tensor,
            edge_hp_height: th.Tensor,
            cc: th.Tensor,):
    '''
    @ edge_cc: [# pair, (2 + 1) + (3 + 1)]
    '''
    
    edges = edge_cc[:, :3]
    ccs = edge_cc[:, 3:]
    
    assert th.all(edges[:, 0] == ccs[:, 0]), ""
    
    edge_id = edges[:, -1]
    cc_id = ccs[:, -1]
    
    curr_edge_hp_normal = edge_hp_normal[edge_id]
    curr_edge_hp_height = edge_hp_height[edge_id]
    curr_cc = cc[cc_id]
    
    # compute distance
    curr_dist = -th.sum(curr_edge_hp_normal * curr_cc, dim=-1, keepdim=True) + curr_edge_hp_height
    
    result = th.cat([edges, ccs, curr_dist], dim=-1)
    
    return result


class WDT(WDT_OURS_S2):
    
    def forward(self, points: th.Tensor, weights: th.Tensor, query_edges: th.Tensor):
        
        '''
        1. Run WDT algorithm
        '''
        tris = self.nond_wdt(points, weights.unsqueeze(-1))
        edges = [tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]
        edges = th.cat(edges, dim=0)
        edges = th.sort(edges, dim=-1)[0]
        edges = th.unique(edges, dim=0)
        
        # cc
        tris_cc = self.th_cc(points, weights, tris)
        
        # divide [query_edges]
        query_edges_exist = tensor_intersect(query_edges, edges)
        query_edges_nexist = tensor_subtract(query_edges, edges)
        
        final_edges = []
        final_edges_probs = []
        
        '''
        2. Evaluate probs for existing edges
        '''
        t_edges_0 = tris[:, [0, 1]]; t_apex_0 = th.arange(len(tris), device=self.device)
        t_edges_1 = tris[:, [1, 2]]; t_apex_1 = th.arange(len(tris), device=self.device)
        t_edges_2 = tris[:, [2, 0]]; t_apex_2 = th.arange(len(tris), device=self.device)
        t_edges = th.cat([t_edges_0, t_edges_1, t_edges_2], dim=0)
        t_apex = th.cat([t_apex_0, t_apex_1, t_apex_2], dim=0)
        
        t_tris = th.cat([t_edges, t_apex.unsqueeze(-1)], dim=-1)
        t_tris[:, :2] = th.sort(t_tris[:, :2], dim=-1)[0]
        t_tris = th.unique(t_tris, dim=0)
        
        _, t_tris_cnt = th.unique(t_tris[:, :2], return_counts=True, dim=0)
        t_tris_cnt_cumsum = th.cumsum(t_tris_cnt, dim=0)
        
        # existing edges that have bounded dual line
        existing_edges_with_probs = None
        for i in range(2):
            if i == 0:
                existing_edges_with_probs = t_tris[t_tris_cnt_cumsum[t_tris_cnt == 2] - 1]
            else:
                add_t = t_tris[t_tris_cnt_cumsum[t_tris_cnt == 2] - 2][:, [-1]]
                existing_edges_with_probs = th.cat([existing_edges_with_probs, add_t], dim=-1)
        # existing edges that have unbounded dual line
        existing_edges_wo_probs = t_tris[t_tris_cnt_cumsum[t_tris_cnt == 1] - 1][:, :2]
        
        '''
        2-1. Evaluate probs for existing edges with bounded dual line
        '''
        curr_edges = existing_edges_with_probs[:, :2]
        curr_edges_endpoint_0 = tris_cc[existing_edges_with_probs[:, 2]]
        curr_edges_endpoint_1 = tris_cc[existing_edges_with_probs[:, 3]]
        curr_edges_dual_point = (curr_edges_endpoint_0 + curr_edges_endpoint_1) * 0.5
        
        # find half planes that are in the current WDT
        hplanes = []
        pairs = [[0, 1], [0, 2], [1, 2], [1, 0], [2, 0], [2, 1]]
        for pair in pairs:
            ch = tris[:, pair]
            hplanes.append(ch)
        hplanes = th.cat(hplanes, dim=0)
        hplanes = th.unique(hplanes, dim=0)
        hplanes_normal, hplanes_height = self.compute_hplane(points, weights, hplanes)
        
        # prepare query edges
        curr_query_edges = []
        pairs = [[0, 1], [1, 0]]
        for pair in pairs:
            qt = curr_edges[:, pair]
            index_col = th.arange(len(curr_edges), device=self.device).unsqueeze(-1)
            qt = th.cat([qt, index_col], dim=-1)
            curr_query_edges.append(qt)
        curr_query_edges = th.cat(curr_query_edges, dim=0)
        curr_query_edges = th.unique(curr_query_edges, dim=0)
        curr_query_edges_cc = curr_edges_dual_point
        
        # prepare half planes
        curr_hplanes = hplanes
        index_col = th.arange(len(hplanes), device=self.device).unsqueeze(-1)
        curr_hplanes = th.cat([hplanes, index_col], dim=-1)
        curr_hplanes_normal = hplanes_normal
        curr_hplanes_height = hplanes_height
        
        curr_callback = lambda x: callback0(x, curr_query_edges_cc, curr_hplanes_normal, curr_hplanes_height)
        with th.no_grad():
            curr_query_edges_val = self.join_indices(
                curr_query_edges,
                curr_hplanes,
                curr_callback
            )
        curr_query_edges_val = curr_query_edges_val[:, [0, 1, 2, -1, 3, 4, 5]] # [tri0, tri1, tri_idx, dist, hp0, hp1, hp_idx]
        curr_query_edges_val = th.unique(curr_query_edges_val, dim=0)   # sort using dist
        _, u_edge_cnt = th.unique(curr_query_edges_val[:, :2], return_counts=True, dim=0)
        u_edge_cnt_cumsum = th.cumsum(u_edge_cnt, dim=0)
        u_edge_cnt_cumsum = th.cat([th.zeros((1,), dtype=th.int64, device=self.device), u_edge_cnt_cumsum[:-1]], dim=0)
        
        # choose minimum
        curr_final_edges = curr_query_edges_val[u_edge_cnt_cumsum][:, :2].to(dtype=th.long)
        curr_final_edges_dist = curr_query_edges_val[u_edge_cnt_cumsum][:, 3]
        curr_final_edges_prob = th.sigmoid(curr_final_edges_dist * 1e3)
        # assert th.all(curr_final_edges_prob >= 0.5), ""
        
        final_edges.append(curr_final_edges)
        final_edges_probs.append(curr_final_edges_prob)
        
        '''
        2-2. Evaluate probs for existing edges with unbounded dual line
        '''
        curr_final_edges = existing_edges_wo_probs
        curr_final_edges_prob = th.ones_like(curr_final_edges[:, 0], dtype=th.float32, device=self.device)
        
        final_edges.append(curr_final_edges)
        final_edges_probs.append(curr_final_edges_prob)
        
        '''
        3. Evaluate probs for non-existing edges
        '''
        curr_edges = query_edges_nexist
        
        # prepare query edges
        curr_query_edges = []
        pairs = [[0, 1], [1, 0]]
        for pair in pairs:
            qt = curr_edges[:, pair]
            curr_query_edges.append(qt)
        curr_query_edges = th.cat(curr_query_edges, dim=0)
        curr_query_edges = th.unique(curr_query_edges, dim=0)
        curr_query_edges = th.cat([curr_query_edges, th.arange(len(curr_query_edges), device=self.device).unsqueeze(-1)], dim=-1)
        curr_edges_hplane_normal, curr_edges_hplane_height = \
            self.compute_hplane(points, weights, curr_query_edges[:, :2])
        
        # prepare tri cc
        curr_tris_cc = []
        pairs = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
        for pair in pairs:
            qt = tris[:, pair]
            index_col = th.arange(len(tris), device=self.device).unsqueeze(-1)
            qt = th.cat([qt, index_col], dim=-1)
            curr_tris_cc.append(qt)
        curr_tris_cc = th.cat(curr_tris_cc, dim=0)
        curr_tris_cc = th.unique(curr_tris_cc, dim=0)
        
        # join
        curr_callback = lambda x: callback1(x, curr_edges_hplane_normal, curr_edges_hplane_height, tris_cc)
        with th.no_grad():
            curr_query_edges_val = self.join_indices(
                curr_query_edges,
                curr_tris_cc,
                curr_callback
            )
        curr_query_edges_val = curr_query_edges_val[:, [0, 1, 2, -1, 3, 4, 5, 6]] # [tri0, tri1, tri_idx, dist, hp0, hp1, hp_idx]
        curr_query_edges_val = th.unique(curr_query_edges_val, dim=0)   # sort using dist
        _, u_edge_cnt = th.unique(curr_query_edges_val[:, :2], return_counts=True, dim=0)
        u_edge_cnt_cumsum = th.cumsum(u_edge_cnt, dim=0)
        u_edge_cnt_cumsum = th.cat([th.zeros((1,), dtype=th.int64, device=self.device), u_edge_cnt_cumsum[:-1]], dim=0)
        
        # choose minimum
        curr_final_edges = curr_query_edges_val[u_edge_cnt_cumsum][:, :2].to(dtype=th.long)
        curr_final_edges_dist = -curr_query_edges_val[u_edge_cnt_cumsum][:, 3]
        curr_final_edges_prob = th.sigmoid(curr_final_edges_dist * 1e3)
        # assert th.all(curr_final_edges_prob <= 0.5), ""
        
        final_edges.append(curr_final_edges)
        final_edges_probs.append(curr_final_edges_prob)
        
        '''
        Aggregate
        '''
        for i in range(len(final_edges)):
            final_edges[i] = th.sort(final_edges[i], dim=-1)[0]
        total_edges = th.cat(final_edges, dim=0)
        total_edges_prob = th.cat(final_edges_probs, dim=0)
        
        total_edges_integ = th.cat([total_edges, total_edges_prob.unsqueeze(-1)], dim=-1)
        total_edges_integ = th.unique(total_edges_integ, dim=0)
        
        # select min prob
        _, total_edges_integ_cnt = th.unique(total_edges_integ[:, :2], return_counts=True, dim=0)
        total_edges_integ_cnt_cumsum = th.cumsum(total_edges_integ_cnt, dim=0)
        total_edges_integ_cnt_cumsum = th.cat([th.zeros((1,), dtype=th.int64, device=self.device), total_edges_integ_cnt_cumsum[:-1]], dim=0)
        
        total_edges_integ = total_edges_integ[total_edges_integ_cnt_cumsum]
        
        total_edges = total_edges_integ[:, :2].to(dtype=th.long)
        total_edges_prob = total_edges_integ[:, 2]
        
        # faces in [faces] but not in [total_faces]
        remain_edges = tensor_subtract(query_edges, total_edges)
        remain_edges_prob = th.full((len(remain_edges),), 0, device=self.device)
        
        total_edges = th.cat([total_edges, remain_edges], dim=0)
        total_edges_prob = th.cat([total_edges_prob, remain_edges_prob], dim=0)
        total_edges_integ = th.cat([total_edges, total_edges_prob.unsqueeze(-1)], dim=-1)
        total_edges_integ = th.unique(total_edges_integ, dim=0)
        
        total_edges = total_edges_integ[:, :2].to(dtype=th.long)
        total_edges_prob = total_edges_integ[:, 2]
        
        assert len(total_edges) == len(query_edges), ""
        
        return total_edges, total_edges_prob