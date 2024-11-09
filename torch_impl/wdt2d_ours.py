import torch as th
from scipy.spatial import ConvexHull as scipyCH
from pytorch3d.ops import knn_points

def tensor_intersect(a: th.Tensor, b: th.Tensor):
    '''
    Get intersection of two tensors of shape [N, D] and [M, D].
    Assume [a] and [b] does not have duplicates in its own.
    '''
    assert a.shape[1] == b.shape[1], "Tensor dimension mismatch"
    merge = th.cat([a, b], dim=0)
    u_merge, u_merge_cnt = th.unique(merge, return_counts=True, dim=0)
    return u_merge[u_merge_cnt > 1]

def tensor_subtract(a: th.Tensor, b: th.Tensor):
    '''
    Subtract common elements in a & b from a.
    Assume [a] and [b] does not have duplicates in its own.
    '''
    assert a.shape[1] == b.shape[1], "Tensor dimension mismatch"
    intersect = tensor_intersect(a, b)
    
    merge_a = th.cat([a, intersect], dim=0)
    u_merge_a, u_merge_cnt_a = th.unique(merge_a, return_counts=True, dim=0)

    return u_merge_a[u_merge_cnt_a == 1]

def callback0(tri_hp: th.Tensor,
                tri_cc: th.Tensor,
                hp_normal: th.Tensor,
                hp_height: th.Tensor):

    '''
    @tri_hp: [# pair, (3 + 1) + (2 + 1)]
    '''
    
    triangles = tri_hp[:, :4]
    hplanes = tri_hp[:, 4:]
    
    assert th.all(triangles[:, 0] == hplanes[:, 0]), ""
    
    tri_id = triangles[:, -1]
    hp_id = hplanes[:, -1]
    
    curr_tri_cc = tri_cc[tri_id]
    curr_hp_normal = hp_normal[hp_id]
    curr_hp_height = hp_height[hp_id]
    
    # compute distance
    curr_dist = -th.sum(curr_hp_normal * curr_tri_cc, dim=-1, keepdim=True) + curr_hp_height
    # curr_dist = th.abs(curr_dist)
    
    # filter
    invalid0 = triangles[:, 1] == hplanes[:, 1]
    invalid1 = triangles[:, 2] == hplanes[:, 1]
    invalid = invalid0 | invalid1
    curr_dist[invalid] = float('inf')
    
    result = th.cat([triangles, hplanes, curr_dist], dim=-1)
    
    return result

class WDT(th.nn.Module):
    
    def __init__(self, device):
        super(WDT, self).__init__()
        self.EPS = 1e-6
        
        self.device = device
        # self.query_triangles_k = 20
        # self.query_triangles = th.empty((0, 3), dtype=th.int64, device=device)
        # self.pointwise_close_half_planes = th.empty((0, 2), dtype=th.int64, device=device)
        
    def nond_wdt(self, points: th.Tensor, weights: th.Tensor):
        
        '''
        Follow convex hull based algorithm of [Aurenhammer 87].
        @ points: shape = [# point, # dim]
        @ weights: shape = [# point, 1]
        '''
        assert points.ndim == 2, ""
        assert weights.ndim == 2, ""

        device = points.device

        # [# point, 1]
        lift = th.sum(points * points, dim=-1, keepdim=True) - (weights)

        # [# point, # dim + 1]
        wpoints = th.cat([points, lift], dim=-1)
        
        chull = scipyCH(wpoints.cpu().numpy())
        
        chull_simplices = th.tensor(chull.simplices, dtype=th.int64, device=device)
        chull_equations = th.tensor(chull.equations, dtype=th.float32, device=device)
        assert len(chull_simplices) == len(chull_equations), ""

        tri_list = chull_simplices[chull_equations[:, -2] <= 0]
        # tri_list = th.tensor(tri_list, dtype=th.int64, device=device)
        tri_list = th.sort(tri_list, dim=-1)[0]
        tri_list = th.unique(tri_list, dim=0)
        
        return tri_list
    
    def find_knn(self, from_pts: th.Tensor, to_pts: th.Tensor, k: int):
        '''
        Find KNN for each point in from_pts to to_pts.
        '''
        knn_result = knn_points(
            from_pts.unsqueeze(0),
            to_pts.unsqueeze(0),
            K=k,
        )
        neighbors = knn_result.idx.squeeze(0)
        
        return neighbors
    
    
    '''
    Joining
    '''

    def combine_tensors(self, 
                        mat0: th.Tensor, 
                        mat1: th.Tensor,
                        mat0_beg: th.Tensor,
                        mat1_unit: th.Tensor):
        
        assert mat0.ndim == 2, ""
        assert mat1.ndim == 2, ""

        # [# item, 1]
        mat0_beg0 = mat0_beg[mat0[:, 0]].unsqueeze(-1)

        # [# item, mat1_unit]
        add = th.arange(0, mat1_unit, dtype=th.int64, device=self.device)
        add = add.unsqueeze(0).expand((mat0_beg0.shape[0], -1))

        # [# item, mat1_unit]
        mat0_collect = mat0_beg0 + add
        # clamp, so that items in [mat0_collect] do not exceed size of [mat1]'s number;
        mat0_collect = th.clamp(mat0_collect, max=len(mat1) - 1)
        
        # [# item * mat1_unit]
        mat0_collect0 = mat0_collect.reshape((-1,))

        # [# item * mat1_unit, # mat1.shape[-1]]
        mat1_collect0 = mat1[mat0_collect0]

        # [# item * mat1_unit, # mat0.shape[-1]]
        mat0_expand = mat0.unsqueeze(1).expand((-1, mat1_unit, -1))
        mat0_expand0 = mat0_expand.reshape((-1, mat0_expand.shape[-1]))

        combination = th.cat([mat0_expand0, mat1_collect0], dim=-1)

        # unique;
        combination = th.unique(combination, dim=0)

        return combination
    
    def join_indices_r(self, 
                       index0: th.Tensor, 
                       index1: th.Tensor, 
                       range0,
                       index0_beg: th.Tensor,
                       index1_cnt: th.Tensor,
                       max_size: int, 
                       callback=None):
        '''
        Recursive function for use in [join_indices].
        '''
        
        w0 = index0.shape[1]
        w1 = index1.shape[1]

        curr_h0_beg = range0[0]
        curr_h0_end = range0[1]
        curr_max_cnt = th.max(index1_cnt[index0[curr_h0_beg:curr_h0_end, 0]])
        curr_size = (curr_h0_end - curr_h0_beg) * curr_max_cnt * (w0 + w1)
        
        curr_index0 = index0[curr_h0_beg:curr_h0_end]
        
        if curr_size > max_size:
            curr_h0_mid = (curr_h0_beg + curr_h0_end) // 2
            curr_joint_index0 = self.join_indices_r(index0, index1, [curr_h0_beg, curr_h0_mid], index0_beg, index1_cnt, max_size, callback)
            curr_joint_index1 = self.join_indices_r(index0, index1, [curr_h0_mid, curr_h0_end], index0_beg, index1_cnt, max_size, callback)
            if len(curr_joint_index0) == 0 and len(curr_joint_index1) > 0:
                return curr_joint_index1
            elif len(curr_joint_index0) > 0 and len(curr_joint_index1) == 0:
                return curr_joint_index0
            elif len(curr_joint_index0) == 0 and len(curr_joint_index1) == 0:
                return curr_joint_index0
            else:
                curr_joint_index2 = th.cat([curr_joint_index0, curr_joint_index1], dim=0)
                return curr_joint_index2
        else:
            # join [curr_t_index0] and [curr_t_index1];
            curr_joint_index = self.combine_tensors(curr_index0, index1, index0_beg, curr_max_cnt)
            curr_joint_index_bool = (curr_joint_index[:, 0] == curr_joint_index[:, w0])
            curr_joint_index0 = curr_joint_index[curr_joint_index_bool]
            if callback is not None and curr_joint_index0.shape[0] > 0:
                curr_joint_index1 = callback(curr_joint_index0)
            else:
                curr_joint_index1 = curr_joint_index0
            return curr_joint_index1
    
    def join_indices(self, index0: th.Tensor, index1: th.Tensor, callback = None):

        '''
        Join two index tensors. 
        e.g.)
        index0 = [[1, 2, 3], [2, 4, 6]]
        index1 = [[1, 3, 4], [2, 5, 7]]
        => [[1, 2, 3, 1, 3, 4], [2, 4, 6, 2, 5, 7]]

        Likewise, indices where the first element is equal are joined.
        Returned indices do not have duplicates and sorted.

        @ callback: Used to postprocess joined indices.
        '''

        assert index0.ndim == 2, ""
        assert index1.ndim == 2, ""

        # sort;
        t_index0 = th.unique(index0, dim=0)
        t_index1 = th.unique(index1, dim=0)

        '''
        Since we are going to match [t_index1]'s first index to [t_index0]'s first index,
        we find corresponding starting index of [t_index1] that is same to [t_index0]'s 
        first index;
        e.g.) t_index0 = [[1, 2, 3],          t_index1 = [[0, 1, 2],
                          [4, 5, 6]]                      [1, 2, 3],
                                                          [1, 4, 5],
                                                          [2, 4, 5],
                                                          [4, 5, 7]]
        then for [1, 2, 3] in t_index0, we should find [1] in t_index1, because first index
        1 starts from t_index1[1]; for [4, 5, 6] in t_index1, we should find [4];
        '''
        
        # since [t_index1[:, 0]] is already sorted...;
        u_t_index1, u_t_index1_cnt = th.unique(t_index1[:, 0], return_counts=True)
        u_t_index1_cnt_cumsum = th.cumsum(u_t_index1_cnt, dim=0)
        u_t_index1_cnt_cumsum = th.cat([th.zeros((1,), dtype=th.int64, device=self.device), u_t_index1_cnt_cumsum[:-1]], dim=0)
        
        # for each first index in [t_index0], find starting point and cnt;
        max_t_index0 = th.max(th.cat([t_index0[:, 0], t_index1[:, 0]], dim=0))
        u_t_index2 = th.zeros((max_t_index0 + 1,), dtype=th.int64, device=self.device)
        u_t_index3 = th.zeros((max_t_index0 + 1,), dtype=th.int64, device=self.device)
        u_t_index2[u_t_index1] = u_t_index1_cnt_cumsum
        u_t_index3[u_t_index1] = u_t_index1_cnt

        # rearrange [t_index0] by count for acceleration;
        t_index0_cnt = u_t_index3[t_index0[:, 0]]
        _, t_index0_cnt_idx = th.sort(t_index0_cnt, stable=True)
        t_index2 = t_index0[t_index0_cnt_idx]

        # estimate required disk space;
        # we require (h0 * h1) * (w0 + w1) size tensor;
        # from that tensor, we only select index pairs that share same first index;
        h0 = t_index0.shape[0]
        MAX_TENSOR_SIZE = 1_000_000_0

        # recursive join;

        joint_indices = self.join_indices_r(t_index2, 
                                            t_index1, 
                                            [0, h0], 
                                            u_t_index2,
                                            u_t_index3,
                                            MAX_TENSOR_SIZE, 
                                            callback)

        return joint_indices
    
    '''
    Circumcenter
    '''
    def th_cc(self,
            positions: th.Tensor,
            weights: th.Tensor,
            tri_idx: th.Tensor):
        
        dimension = positions.shape[1]

        '''
        1. Gather point coordinates for each simplex.
        '''
        num_simplex = tri_idx.shape[0]

        # [# simplex, # dim + 1, # dim]
        simplex_points = positions[tri_idx]

        # [# simplex, # dim + 1]
        simplex_weights = weights[tri_idx]

        '''
        2. Change points in [# dim] dimension to hyperplanes in [# dim + 1] dimension
        '''
        # [# simplex, # dim + 1, # dim + 2]
        hyperplanes0 = th.ones_like(simplex_points[:, :, [0]]) * -1.
        hyperplanes1 = simplex_weights.unsqueeze(-1) - \
            th.sum(simplex_points * simplex_points, dim=-1, keepdim=True)
        hyperplanes = th.cat([simplex_points * 2., hyperplanes0, hyperplanes1], dim=-1)

        '''
        3. Find intersection of hyperplanes above to get circumcenter.
        '''
        # @TODO: Speedup by staying on original dimension...
        mats = []
        for dim in range(dimension + 2):
            cols = list(range(dimension + 2))
            cols = cols[:dim] + cols[(dim + 1):]

            # [# simplex, # dim + 1, # dim + 1]
            mat = hyperplanes[:, :, cols]
            mats.append(mat)

        # [# simplex * (# dim + 2), # dim + 1, # dim + 1]
        detmat = th.cat(mats, dim=0)

        # [# simplex * (# dim + 2)]
        det = th.det(detmat)

        # [# simplex, # dim + 2]
        hyperplane_intersections0 = det.reshape((dimension + 2, num_simplex))
        hyperplane_intersections0 = th.transpose(hyperplane_intersections0.clone(), 0, 1)
        sign = 1.
        for dim in range(dimension + 2):
            hyperplane_intersections0[:, dim] = hyperplane_intersections0[:, dim] * sign
            sign *= -1.
            
        # [# simplex, # dim + 2]
        eps = 1e-6
        last_dim = hyperplane_intersections0[:, [-1]]
        last_dim = th.sign(last_dim) * th.clamp(th.abs(last_dim), min=eps)
        last_dim = th.where(last_dim == 0., th.ones_like(last_dim) * eps, last_dim)
        hyperplane_intersections = hyperplane_intersections0[:, :] / \
                                        last_dim

        '''
        Projection
        '''
        # [# tri, # dim]

        circumcenters = hyperplane_intersections[:, :-2]
        if th.any(th.isnan(circumcenters)) or th.any(th.isinf(circumcenters)):
            raise ValueError()

        return circumcenters
    
    '''
    Half plane
    '''
    def compute_hplane(self, points: th.Tensor, weights: th.Tensor, hplanes: th.Tensor):
        '''
        Compute 1-flats defined in [hplanes].
        1-flats are hyperplanes that are defined by two points.
        Each 1-flat is defined as a (unit) normal vector and height from the origin.

        @ points: [# point, # dim]
        @ weights: [# point, 1]
        @ hplanes: [# flat, 2]
        '''

        assert hplanes.ndim == 2 and hplanes.shape[1] == 2, ""

        findices = hplanes

        # [# flat, # dim]
        points0 = points[findices[:, 0]]
        points1 = points[findices[:, 1]]
        weights0 = weights[findices[:, 0]]
        weights1 = weights[findices[:, 1]]

        # [# flat, 1]
        points_diff = th.norm(points1 - points0, p=2, dim=-1, keepdim=True)
        points_diff = th.clamp(points_diff * points_diff, min=self.EPS)
        weights_diff = (weights0 - weights1) # - (weights1 * weights1)

        alphas = 0.5 - (weights_diff.unsqueeze(-1) / (2. * points_diff))

        mid_points = (alphas * points0) + ((1. - alphas) * points1)

        flat_normals = th.nn.functional.normalize(points1 - points0, p=2, dim=-1, eps=self.EPS)
        flat_heights = th.sum(flat_normals * mid_points, dim=-1, keepdim=True)

        return flat_normals, flat_heights
        
    def forward(self, points: th.Tensor, weights: th.Tensor, faces: th.Tensor):
        
        '''
        1. Run WDT algorithm
        '''
        tris = self.nond_wdt(points, weights.unsqueeze(-1))
        
        # divide [query_faces]
        query_triangles_exist = tensor_intersect(faces, tris)
        query_triangles_nexist = tensor_subtract(faces, tris)
        
        '''
        2-2. Compute dual points for query triangles
        '''
        entire_query_triangles = th.cat([query_triangles_exist, query_triangles_nexist], dim=0)
        entire_query_triangles_cc = self.th_cc(points, weights, entire_query_triangles)
        
        # [# tri, 2]
        query_triangles_exist_cc = entire_query_triangles_cc[:len(query_triangles_exist)]
        query_triangles_nexist_cc = entire_query_triangles_cc[len(query_triangles_exist):]
        
        '''
        3. Compute half planes to consider: half planes that comprise PD
        '''
        
        # find half planes that are in the current WDT
        curr_hplanes = []
        pairs = [[0, 1], [0, 2], [1, 2], [1, 0], [2, 0], [2, 1]]
        for pair in pairs:
            ch = tris[:, pair]
            curr_hplanes.append(ch)
        curr_hplanes = th.cat(curr_hplanes, dim=0)
        curr_hplanes = th.unique(curr_hplanes, dim=0)
        
        hplanes = curr_hplanes
        hplanes_normal, hplanes_height = self.compute_hplane(points, weights, hplanes)
        
        '''
        4. Compute probabilities for existing triangles
        (query_triangles_exist)
        '''
        # prepare query triangles
        curr_query_triangles = []
        pairs = [[0, 1, 2], [1, 0, 2], [2, 0, 1]]
        for pair in pairs:
            qt = query_triangles_exist[:, pair]
            index_col = th.arange(len(query_triangles_exist), device=self.device).unsqueeze(-1)
            qt = th.cat([qt, index_col], dim=-1)
            curr_query_triangles.append(qt)
        curr_query_triangles = th.cat(curr_query_triangles, dim=0)
        curr_query_triangles = th.unique(curr_query_triangles, dim=0)
        curr_query_triangles_cc = query_triangles_exist_cc
        
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
        e_faces = curr_query_triangles_val[u_tri_cnt_cumsum][:, :3].to(dtype=th.long)
        e_faces_dist = curr_query_triangles_val[u_tri_cnt_cumsum][:, 4]
        e_faces_prob = th.sigmoid(e_faces_dist * 1e3)
        # assert th.all(e_faces_prob >= 0.5), ""
        
        '''
        5. Compute probabilities for non-existing triangles
        (query_triangles_nexist)
        '''
        # prepare query triangles
        curr_query_triangles = []
        pairs = [[0, 1, 2], [1, 0, 2], [2, 0, 1]]
        for pair in pairs:
            qt = query_triangles_nexist[:, pair]
            index_col = th.arange(len(query_triangles_exist), device=self.device).unsqueeze(-1)
            qt = th.cat([qt, index_col], dim=-1)
            curr_query_triangles.append(qt)
        curr_query_triangles = th.cat(curr_query_triangles, dim=0)
        curr_query_triangles = th.unique(curr_query_triangles, dim=0)
        curr_query_triangles_cc = query_triangles_nexist_cc
        
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
        ne_faces = curr_query_triangles_val[u_tri_cnt_cumsum][:, :3].to(dtype=th.long)
        ne_faces_dist = curr_query_triangles_val[u_tri_cnt_cumsum][:, 4]
        ne_faces_prob = th.sigmoid(ne_faces_dist * 1e3)
        # assert th.all(ne_faces_prob <= 0.5), ""
        
        '''
        Aggregate
        '''
        e_faces = th.sort(e_faces, dim=-1)[0]
        ne_faces = th.sort(ne_faces, dim=-1)[0]
        
        total_faces = th.cat([e_faces, ne_faces], dim=0)
        total_faces_prob = th.cat([e_faces_prob, ne_faces_prob], dim=0)
        
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
    