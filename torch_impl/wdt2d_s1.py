import torch as th
from torch_impl.wdt2d import WDT as WDT2

class WDT1(WDT2):
    
    def process(self, 
                points: th.Tensor,
                weights: th.Tensor,
                neighbors: th.Tensor, 
                neighbor_points: th.Tensor,
                neighbor_weights: th.Tensor,
                normals: th.Tensor):
        
        device = points.device
        
        num_points = len(points)
        
        nn_size = neighbors.shape[1] - 1
        
        tmp_neighbors = neighbors.unsqueeze(-1).expand([-1, -1, 3])
        nn_coords = neighbor_points[:, 1:]       # exclude itself;
        nn_weights = neighbor_weights[:, 1:]
        
        centers = neighbor_points[:, 0]
        centers_weights = neighbor_weights[:, 0]
        
        # [compute_triangles_local_geodesic_distances] of [weighted_triangulation2D]
        
        nn_coords = nn_coords[:, :, :2]
        centers = centers[:, :2]
        
        # [half_planes[:, :, :2]] is the directional vector for planes;
        # [half_planes[:, :, 2]] is the location of the middle point on the half plane;
        # half_planes.shape = [# points, # neighbors, 3]
    
        half_planes = self.compute_half_planes(nn_coords, nn_weights, centers, centers_weights)
        
        # find sample points for all half planes for center points;
        # intersections.shape = [# point, # neighbors, 3]

        intersections = self.compute_half_plane_intersections(centers, half_planes)
        intersections = th.cat([intersections, th.ones_like(intersections[:, :, [0]])], dim=-1)
        
        # compute the distance between the intersection points (N points) and the half planes (N)
        # N = number of neighbors, do inner product to get distances;
        # [inter_dist00] shape = [# points, N, N]
        inter_dist00 = th.sum(half_planes.unsqueeze(1).expand([-1, intersections.shape[1], -1, -1]) * \
                                intersections.unsqueeze(2).expand([-1, -1, half_planes.shape[1], -1]), dim=-1)
        
        # [# point, # neighbors, 1]
        intersection_pairs = th.arange(intersections.shape[1]).unsqueeze(0).expand([num_points, -1]).unsqueeze(-1)
        intersection_pairs = intersection_pairs.to(device=device)
        
        # for each intersection point of a certain center point, we find out which half planes
        # contributed to the intersection point and condense the information in [index_couples];
        # index_pairs.shape = [# points, # intersection points, 1, 3]
        # e.g. index_pairs[i, j, k]: k'th hyperplane among 1 hyperplanes that generated
        # j-th intersection point of i-th center points;
        index_pairs_a = th.arange(num_points).unsqueeze(-1).unsqueeze(-1).expand([-1, intersections.shape[1], 1]).to(device=device)
        index_pairs_b = th.arange(intersections.shape[1]).unsqueeze(0).unsqueeze(-1).expand([num_points, -1, 1]).to(device=device)
        index_pairs = th.stack([index_pairs_a, index_pairs_b, intersection_pairs], dim=-1)
        
        # for each triangle we want to ignore the current couple to compute the distance to a "virtual" voronoi cell
        # @sanghyun: we basically compute distance from intersection points, which could be circumcenters of a DT,
        # to the edges of virtual voronoi cell and determine if the intersection points are in the cell or not.
        # since virtual voronoi cell does not consider half planes generated between center point and those points that
        # comprise half planes that make up the intersection points, we designate such indices here;
        # [to_ignore] shape = [# center point, # intersection point, # half plane]
        index_pairs_0 = index_pairs[:, :, :, 0].reshape(-1)
        index_pairs_1 = index_pairs[:, :, :, 1].reshape(-1)
        index_pairs_2 = index_pairs[:, :, :, 2].reshape(-1)
        to_ignore = th.zeros_like(inter_dist00, dtype=th.bool)
        to_ignore[(index_pairs_0, index_pairs_1, index_pairs_2)] = True
        
        inter_dist0 = th.where(to_ignore, -th.ones_like(inter_dist00) * 1e10, inter_dist00)
        # inter_dist = th.where(th.abs(inter_dist0) < self.EPS, -th.ones_like(inter_dist0) * 1e10, inter_dist0)
        
        # get exact DTs;
        is_dt_discrete = self.discrete_dt(inter_dist0)
        
        # get smooth DTs, note that we use [inter_dist0], not [inter_dist];
        is_dt_smooth = self.smooth_dt(inter_dist0)
        
        # now gather global point indices for each edge;
        # neighbors.shape = [# point, # nn_size] / intersection_pairs.shape = [# point, # intersection points, 2]
        # tmp_neighbors.shape = [# point, # intersection points, # nn_size] 
        tmp_neighbors = neighbors[:, 1:].unsqueeze(1).expand([-1, intersection_pairs.shape[1], -1])
        tri_global_neighbor_indices = th.gather(tmp_neighbors, dim=-1, index=intersection_pairs)
        tri_first_indices = th.arange(num_points).unsqueeze(-1).unsqueeze(-1).expand([-1, tri_global_neighbor_indices.shape[1], 1]).to(device=device)
        tri_global_indices = th.cat([tri_first_indices, tri_global_neighbor_indices], axis=-1)
        
        return is_dt_discrete, is_dt_smooth, tri_global_indices
     
    def postprocess(self,
                    points: th.Tensor,
                    weights: th.Tensor,
                    is_dt_discrete: th.Tensor,
                    is_dt_smooth: th.Tensor,
                    dt_indices: th.Tensor):
        
        is_dt_discrete = is_dt_discrete.reshape([-1])
        is_dt_smooth = is_dt_smooth.reshape([-1])
        dt_indices = dt_indices.reshape([-1, 2])
        
        # cull out triangles with too small scores;
        
        valid_indices = th.where(is_dt_smooth > 1e-5)
        is_dt_discrete = is_dt_discrete[valid_indices]
        is_dt_smooth = is_dt_smooth[valid_indices]
        dt_indices = dt_indices[valid_indices]
        
        # only select unique indices in [dt_indices];
        
        num_points = len(points)
        
        # convert indices into single value;
        
        n_dt_indices = th.sort(dt_indices, dim=-1)[0]
        n_dt_indices = n_dt_indices[:, 0] * num_points + \
                        n_dt_indices[:, 1]
        
        n_dt_indices, n_dt_indices0 = th.sort(n_dt_indices)
        n_is_dt_discrete = is_dt_discrete[n_dt_indices0]
        n_is_dt_smooth = is_dt_smooth[n_dt_indices0]
        n_dt_indices00 = dt_indices[n_dt_indices0]
        
        _, n_dt_indices_cnt = th.unique(n_dt_indices, return_counts=True)
        n_dt_indices0 = th.cumsum(n_dt_indices_cnt, dim=-1) - 1
        
        n_is_dt_discrete = n_is_dt_discrete[n_dt_indices0]
        n_is_dt_smooth = n_is_dt_smooth[n_dt_indices0]
        n_dt_indices = n_dt_indices00[n_dt_indices0]
                        
        return n_is_dt_discrete, n_is_dt_smooth, n_dt_indices
       
    def compute_half_plane_intersections(self, 
                                        centers: th.Tensor,
                                        half_planes: th.Tensor):
        
        '''
        @ centers: Tensor of shape [# points, 2].
        @ half_planes: Tensor of shape [# points, # neighbors, 3].
        
        Compute sample point on each half plane by projecting
        centers on the half planes.
        '''
        
        ha = half_planes[:, :, 0]   # [# points, # neighbors]
        hb = half_planes[:, :, 1]
        hc = half_planes[:, :, 2]
        
        px = centers[:, [0]]    # [# points, 1]
        py = centers[:, [1]]    # [# points, 1]
        
        fx = (-ha * hc) + (-ha * hb * py) + (hb * hb * px)
        fx = fx / (ha * ha + hb * hb)
        
        fy = (-hb * hc) + (-ha * hb * px) + (ha * ha * py)
        fy = fy / (ha * ha + hb * hb)   # [# points, # neighbors]
        
        # sample_points.shape = [# points, # neighbors, 2]
        sample_points = th.cat([fx.unsqueeze(-1), fy.unsqueeze(-1)], dim=-1)
        
        return sample_points