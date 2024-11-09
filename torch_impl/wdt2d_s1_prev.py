import torch as th
from torch_impl.wdt2d_ours import WDT as WDT_OURS_S2
from torch_impl.wdt2d_ours import tensor_intersect, tensor_subtract

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

class WDT(WDT_OURS_S2):
    
    def th_cc(self, points: th.Tensor, weights: th.Tensor, edges: th.Tensor):
        '''
        Sample points on dual form of the given edges by projection
        '''
        
        hp_normals, hp_heights = self.compute_hplane(points, weights, edges)
        
        ha = hp_normals[:, 0]   # [# edges]
        hb = hp_normals[:, 1]
        hc = hp_heights.squeeze(-1)
        
        px = points[edges[:, 0]][:, 0]    # [# edges,]
        py = points[edges[:, 0]][:, 1]    # [# edges,]
        
        fx = (-ha * hc) + (-ha * hb * py) + (hb * hb * px)
        fx = fx / (ha * ha + hb * hb)
        
        fy = (-hb * hc) + (-ha * hb * px) + (ha * ha * py)
        fy = fy / (ha * ha + hb * hb)   # [# edges]
        
        # sample_points.shape = [# edge, 2]
        sample_points = th.cat([fx.unsqueeze(-1), fy.unsqueeze(-1)], dim=-1)
        
        return sample_points
    
    def forward(self, points: th.Tensor, weights: th.Tensor, query_edges: th.Tensor):
        
        '''
        2-2. Compute dual points for query edges
        '''
        entire_query_edges = query_edges
        entire_query_edges_cc = self.th_cc(points, weights, entire_query_edges)
        
        '''
        3. Compute half planes to consider: all half planes
        '''
        
        # find every half plane
        curr_hplanes = th.combinations(th.arange(len(points), device=self.device, dtype=th.long), 2)
        curr_hplanes = curr_hplanes.to(dtype=th.long)
        curr_hplanes = [curr_hplanes, curr_hplanes[:, [1, 0]]]
        curr_hplanes = th.cat(curr_hplanes, dim=0)
        curr_hplanes = th.unique(curr_hplanes, dim=0)
        
        hplanes = curr_hplanes
        hplanes_normal, hplanes_height = self.compute_hplane(points, weights, hplanes)
        
        '''
        4. Compute probabilities for edges
        '''
        # prepare query edges
        curr_query_edges = []
        pairs = [[0, 1], [1, 0]]
        for pair in pairs:
            qt = entire_query_edges[:, pair]
            index_col = th.arange(len(entire_query_edges), device=self.device).unsqueeze(-1)
            qt = th.cat([qt, index_col], dim=-1)
            curr_query_edges.append(qt)
        curr_query_edges = th.cat(curr_query_edges, dim=0)
        curr_query_edges = th.unique(curr_query_edges, dim=0)
        curr_query_edges_cc = entire_query_edges_cc
        
        # prepare half planes
        curr_hplanes = hplanes
        index_col = th.arange(len(hplanes), device=self.device).unsqueeze(-1)
        curr_hplanes = th.cat([curr_hplanes, index_col], dim=-1)
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
        
        final_edges = [curr_final_edges]
        final_edges_probs = [curr_final_edges_prob]
        
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