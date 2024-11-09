import torch as th

from torch_impl.wdt2d_ours import WDT as WDT_OURS
from torch_impl.wdt2d_ours import tensor_intersect, tensor_subtract, callback0

class WDT(WDT_OURS):
    
    def forward(self, points: th.Tensor, weights: th.Tensor, faces: th.Tensor):
        
        '''
        2-2. Compute dual points for query triangles
        '''
        entire_query_triangles = faces
        entire_query_triangles_cc = self.th_cc(points, weights, entire_query_triangles)
        
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
        4. Compute probabilities for triangles
        '''
        # prepare query triangles
        curr_query_triangles = []
        pairs = [[0, 1, 2], [1, 0, 2], [2, 0, 1]]
        for pair in pairs:
            qt = entire_query_triangles[:, pair]
            index_col = th.arange(len(entire_query_triangles), device=self.device).unsqueeze(-1)
            qt = th.cat([qt, index_col], dim=-1)
            curr_query_triangles.append(qt)
        curr_query_triangles = th.cat(curr_query_triangles, dim=0)
        curr_query_triangles = th.unique(curr_query_triangles, dim=0)
        curr_query_triangles_cc = entire_query_triangles_cc
        
        # prepare half planes
        curr_hplanes = hplanes
        index_col = th.arange(len(hplanes), device=self.device).unsqueeze(-1)
        curr_hplanes = th.cat([curr_hplanes, index_col], dim=-1)
        curr_hplanes_normal = hplanes_normal
        curr_hplanes_height = hplanes_height
        
        curr_callback = lambda x: callback0(x, curr_query_triangles_cc, curr_hplanes_normal, curr_hplanes_height)
        with th.no_grad():
            curr_query_triangles_val = self.join_indices(
                curr_query_triangles,
                curr_hplanes,
                curr_callback
            )
        curr_query_triangles_val = curr_query_triangles_val[:, [0, 1, 2, 3, -1, 4, 5, 6]] # [tri0, tri1, tri2, tri_idx, dist, hp0, hp1, hp_idx]
        curr_query_triangles_val = th.unique(curr_query_triangles_val, dim=0)   # sort using dist
        _, u_tri_cnt = th.unique(curr_query_triangles_val[:, :3], return_counts=True, dim=0)
        u_tri_cnt_cumsum = th.cumsum(u_tri_cnt, dim=0)
        u_tri_cnt_cumsum = th.cat([th.zeros((1,), dtype=th.int64, device=self.device), u_tri_cnt_cumsum[:-1]], dim=0)
        
        # choose minimum
        final_faces = curr_query_triangles_val[u_tri_cnt_cumsum][:, :3].to(dtype=th.long)
        final_faces_dist = curr_query_triangles_val[u_tri_cnt_cumsum][:, 4]
        final_faces_prob = th.sigmoid(final_faces_dist * 1e3)
        
        '''
        Aggregate
        '''
        final_faces = th.sort(final_faces, dim=-1)[0]
        
        total_faces = final_faces # th.cat([e_faces, ne_faces], dim=0)
        total_faces_prob = final_faces_prob # th.cat([e_faces_prob, ne_faces_prob], dim=0)
        
        total_faces_integ = th.cat([total_faces, total_faces_prob.unsqueeze(-1)], dim=-1)
        total_faces_integ = th.unique(total_faces_integ, dim=0)
        
        # select min prob
        _, total_faces_integ_cnt = th.unique(total_faces_integ[:, :3], return_counts=True, dim=0)
        total_faces_integ_cnt_cumsum = th.cumsum(total_faces_integ_cnt, dim=0)
        total_faces_integ_cnt_cumsum = th.cat([th.zeros((1,), dtype=th.int64, device=self.device), total_faces_integ_cnt_cumsum[:-1]], dim=0)
        
        total_faces_integ = total_faces_integ[total_faces_integ_cnt_cumsum]
        
        total_faces = total_faces_integ[:, :3].to(dtype=th.long)
        total_faces_prob = total_faces_integ[:, 3]
        
        # faces in [faces] but not in [total_faces]
        remain_faces = tensor_subtract(faces, total_faces)
        remain_faces_prob = th.full((len(remain_faces),), 0, device=self.device)
        
        total_faces = th.cat([total_faces, remain_faces], dim=0)
        total_faces_prob = th.cat([total_faces_prob, remain_faces_prob], dim=0)
        total_faces_integ = th.cat([total_faces, total_faces_prob.unsqueeze(-1)], dim=-1)
        total_faces_integ = th.unique(total_faces_integ, dim=0)
        
        total_faces = total_faces_integ[:, :3].to(dtype=th.long)
        total_faces_prob = total_faces_integ[:, 3]
        
        assert len(total_faces) == len(faces), ""
        
        return total_faces, total_faces_prob