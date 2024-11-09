import torch as th
from pytorch3d.ops import knn_points

class WDT(th.nn.Module):
    
    def __init__(self, num_trigs: int, nn_size: int=-1):
        # @nn_size: Number of nearest neighbors to find;
        
        super(WDT, self).__init__()
        self.num_trigs = num_trigs
        self.nn_size = nn_size
        self.EPS = 1e-6
        
    def forward(self, points: th.Tensor, weights: th.Tensor):
        
        # neighbors = [# point, # nn_size] (indices of nearest neighbors)
        # neighbor_points = [# point, # nn_size, 3]
        # neighbor_weights = [# point, # nn_size]
        # normals = [# point, 3]
        neighbors, neighbor_points, neighbor_weights, normals = \
            self.preprocess(points, weights)
            
        is_dt_discrete, is_dt_smooth, dt_indices = \
            self.process(points, weights, neighbors, neighbor_points, neighbor_weights, normals)
        
        is_dt_discrete, is_dt_smooth, dt_indices = \
            self.postprocess(points, weights, is_dt_discrete, is_dt_smooth, dt_indices)
        
        return is_dt_discrete, is_dt_smooth, dt_indices
    
    def preprocess(self, points: th.Tensor, weights: th.Tensor):
        
        # points = [# point, 2]
        # weights = [# point]
        
        # find nearest neighbors;
        
        num_points = len(points)
        
        # distances = [# point, # point]
        # distances = (points.unsqueeze(1).expand((-1, num_points, -1)) - \
        #                 points.unsqueeze(0).expand((num_points, -1, -1))).norm(p=2, dim=-1)
        
        nn_size = num_points if self.nn_size == -1 else self.nn_size + 1
        if nn_size > num_points:
            nn_size = num_points
        
        # neighbors = [# point, # nn_size]
        # _, neighbors = th.topk(distances, k=nn_size, dim=-1, largest=False,)
        
        knn_result = knn_points(
            points.unsqueeze(0),
            points.unsqueeze(0),
            K=nn_size,
        )
        neighbors = knn_result.idx.squeeze(0)
        
        tmp_points = points.unsqueeze(0).expand((num_points, -1, -1))
        tmp_neighbors = neighbors.unsqueeze(-1).expand((-1, -1, 2))
        neighbor_points = th.gather(tmp_points, dim=1, index=tmp_neighbors)
        neighbor_points[:, :] -= neighbor_points[:, [0]]
        neighbor_points = th.cat([neighbor_points, th.zeros_like(neighbor_points)[:, :, [0]]], dim=-1)
        
        tmp_weights = weights.unsqueeze(0).expand((num_points, -1))
        neighbor_weights = th.gather(tmp_weights, dim=1, index=neighbors)
        
        normals = th.zeros((num_points, 3), dtype=th.float32, device=points.device)
        normals[:, 2] = 1.
        
        return neighbors, neighbor_points, neighbor_weights, normals
    
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
        pairs = self.index_pairs(nn_size, points.device)
        
        tmp_normals = normals.unsqueeze(0).expand([num_points, -1, -1])
        tmp_neighbors = neighbors.unsqueeze(-1).expand([-1, -1, 3])
        
        nn_coords = neighbor_points[:, 1:]       # exclude itself;
        nn_coords_normal = th.gather(tmp_normals, 1, tmp_neighbors)[:, 1:]
        nn_weights = neighbor_weights[:, 1:]
        
        centers = neighbor_points[:, 0]
        centers_weights = neighbor_weights[:, 0]
        centers_normal = normals[:len(centers)]
        
        # [compute_triangles_local_geodesic_distances] of [weighted_triangulation2D]
        
        n_neighbors = nn_coords.shape[1]
        
        nn_coords = nn_coords[:, :, :2]
        centers = centers[:, :2]
        
        # [half_planes[:, :, :2]] is the directional vector for planes;
        # [half_planes[:, :, 2]] is the location of the middle point on the half plane;
        # half_planes.shape = [# points, # neighbors, 3]
    
        half_planes = self.compute_half_planes(nn_coords, nn_weights, centers, centers_weights)
        
        # find intersection points for all half plane pairs for center points;
        # intersections.shape = [# point, # pairs, 3] (valid intersection if [x, y, 0] is 1.);
        
        intersections = self.compute_half_plane_intersections(half_planes, pairs)
        
        # TODO couples should be half
        # [direction] is the direction from center points to intersection points;
        
        i_direction = intersections[:, :, :2] - centers.unsqueeze(1)
        
        # boolean tensor that is true if direction falls in to some threshold;
        
        i_direction1 = th.logical_and(i_direction[:, :, 0] < 0., i_direction[:, :, 1] < 0.)
        i_direction2 = th.logical_and(i_direction[:, :, 0] < 0., i_direction[:, :, 1] > 0.)
        i_direction3 = th.logical_and(i_direction[:, :, 0] > 0., i_direction[:, :, 1] < 0.)
        i_direction4 = th.logical_and(i_direction[:, :, 0] > 0., i_direction[:, :, 1] > 0.)
        
        # [distances] is the distance from center points to intersection points;
        # distances.shape = [# points, # pairs]
        
        distances = th.sum(th.square(i_direction), dim=-1)
        
        # we are going to select [n_trigs] number of edges for constructing voronoi cell;
        # relax this problem by considering distance to the intersection points rather than edges (half planes);
        # also, we divide directions in 4 and select top most half planes in each direction (not so accurate, but works);
        # shape = [# points, # topk]
        _, closest_intersections_idx1 = th.topk(th.where(i_direction1, distances, distances * 1000), k=self.num_trigs // 4, dim=-1, largest=False)
        _, closest_intersections_idx2 = th.topk(th.where(i_direction2, distances, distances * 1000), k=self.num_trigs // 4, dim=-1, largest=False)
        _, closest_intersections_idx3 = th.topk(th.where(i_direction3, distances, distances * 1000), k=self.num_trigs // 4, dim=-1, largest=False)
        _, closest_intersections_idx4 = th.topk(th.where(i_direction4, distances, distances * 1000), k=self.num_trigs // 4, dim=-1, largest=False)
        
        # indices of intersection points that are closest to the center points (there could be duplicate, because direction_i could be all False for some points);
        # shape = [# points, # topk * 4]
        closest_intersections_idx = th.cat([closest_intersections_idx1, closest_intersections_idx2, closest_intersections_idx3, closest_intersections_idx4], axis=1)
        
        # @ test code to test every intersection point;
        # closest_intersections_idx = th.arange(intersections.shape[1], dtype=th.long, device=device).unsqueeze(0).expand([num_points, -1])
        # self.num_trigs = closest_intersections_idx.shape[1]
        
        # for each intersection point, decide indices of half planes related to it;
        # e_pairs.shape = [# points, # pairs, 2]
        # e_closest_intersections_idx.shape = [# points, # topk * 4, 2]
        e_pairs = pairs.unsqueeze(0).expand([num_points, -1, -1])
        e_closest_intersections_idx = closest_intersections_idx.unsqueeze(-1).expand([-1, -1, 2])
        
        intersection_pairs = th.gather(e_pairs, dim=1, index=e_closest_intersections_idx)
        
        # intersections.shape = [# points, # topk * 4, 3]
        e_closest_intersections_idx = closest_intersections_idx.unsqueeze(-1).expand([-1, -1, 3])
        intersections = th.gather(intersections, dim=1, index=e_closest_intersections_idx)
        
        # compute the distance between the intersection points (N**2 points) and the half planes (N)
        # N = number of neighbors, do inner product to get distances;
        # [inter_dist00] shape = [# points, # topk * 4 (= # intersection points), # half plane (= # neighbors)]
        inter_dist00 = th.sum(half_planes.unsqueeze(1).expand([-1, intersections.shape[1], -1, -1]) * \
                                intersections.unsqueeze(2).expand([-1, -1, half_planes.shape[1], -1]), dim=-1)
        
        # if [intersections] were invalid, make these distances arbitrarily large;
        inter_dist00 = th.where((intersections[:, :, [0]] > 1e6).expand([-1, -1, inter_dist00.shape[2]]),
                                th.ones_like(inter_dist00) * 1e6,
                                inter_dist00)
        
        # for each intersection point of a certain center point, we find out which half planes
        # contributed to the intersection point and condense the information in [index_couples];
        # index_pairs.shape = [# points, # intersection points, 2, 3]
        # e.g. index_pairs[i, j, k]: k'th hyperplane among 2 hyperplanes that generated
        # j-th intersection point of i-th center points;
        index_pairs_a = th.arange(num_points).unsqueeze(-1).unsqueeze(-1).expand([-1, self.num_trigs, 2]).to(device=device)
        index_pairs_b = th.arange(self.num_trigs).unsqueeze(0).unsqueeze(-1).expand([num_points, -1, 2]).to(device=device)
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
        #inter_dist = th.where(th.abs(inter_dist0) < self.EPS, -th.ones_like(inter_dist0) * 1e10, inter_dist0)
        
        # get exact DTs;
        is_dt_discrete = self.discrete_dt(inter_dist0)
        
        # get smooth DTs, note that we use [inter_dist0], not [inter_dist];
        is_dt_smooth = self.smooth_dt(inter_dist0)
        
        # now gather global point indices for each triangle;
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
        dt_indices = dt_indices.reshape([-1, 3])
        
        # cull out triangles with too small scores;
        
        valid_indices = th.where(is_dt_smooth > 1e-5)
        is_dt_discrete = is_dt_discrete[valid_indices]
        is_dt_smooth = is_dt_smooth[valid_indices]
        dt_indices = dt_indices[valid_indices]
        
        # only select unique indices in [dt_indices];
        
        num_points = len(points)
        
        # convert indices into single value;
        
        n_dt_indices = th.sort(dt_indices, dim=-1)[0]
        n_dt_indices = n_dt_indices[:, 0] * num_points * num_points + \
                        n_dt_indices[:, 1] * num_points + \
                        n_dt_indices[:, 2]
        
        n_dt_indices, n_dt_indices0 = th.sort(n_dt_indices)
        n_is_dt_discrete = is_dt_discrete[n_dt_indices0]
        n_is_dt_smooth = is_dt_smooth[n_dt_indices0]
        n_dt_indices00 = dt_indices[n_dt_indices0]
        
        _, n_dt_indices_cnt = th.unique(n_dt_indices, return_counts=True)
        n_dt_indices0 = th.cumsum(n_dt_indices_cnt, dim=-1) - 1
        
        n_is_dt_discrete = n_is_dt_discrete[n_dt_indices0]
        n_is_dt_smooth = n_is_dt_smooth[n_dt_indices0]
        n_dt_indices = n_dt_indices00[n_dt_indices0]
                        
        # is_dt_discrete_list = []
        # is_dt_smooth_list = []
        # dt_indices_list = []
        # added_dt_indices = []
        
        # for i, ndi in enumerate(n_dt_indices):
            
        #     cndi = ndi.cpu().item()
        #     if cndi in added_dt_indices:
        #         continue
            
        #     added_dt_indices.append(cndi)
            
        #     c_is_dt_discrete = is_dt_discrete[i]
        #     c_is_dt_smooth = is_dt_smooth[i]
        #     c_dt_indices = dt_indices[i]
            
        #     # cull out triangles that have too low score;
        #     if c_is_dt_discrete < 1e-5 and c_is_dt_smooth < 1e-5:
        #         continue
            
        #     is_dt_discrete_list.append(c_is_dt_discrete)
        #     is_dt_smooth_list.append(c_is_dt_smooth)
        #     dt_indices_list.append(c_dt_indices)
        
        # n_is_dt_discrete = th.stack(is_dt_discrete_list)
        # n_is_dt_smooth = th.stack(is_dt_smooth_list)
        # n_dt_indices = th.stack(dt_indices_list, dim=0)
        
        return n_is_dt_discrete, n_is_dt_smooth, n_dt_indices
       
    def discrete_dt(self, inter_dist: th.Tensor):
        
        # @inter_dist: Tensor of shape [# points, # intersection points, # half planes].
        # To be a circumcenter of a valid DT, the signed distances to the all half planes
        # from an intersection point should be negative.
        
        inter_dist = -th.sign(inter_dist)
        is_triangle = th.sum(inter_dist, dim=-1)
        is_triangle = th.where(is_triangle < inter_dist.shape[2], 
                               th.zeros_like(is_triangle), 
                               th.ones_like(is_triangle))
        return is_triangle
    
    def smooth_dt(self, inter_dist: th.Tensor):
        
        # @inter_dist: Tensor of shape [# points, # intersection points, # half planes].
        # To be a circumcenter of a valid DT, the signed distances to the all half planes
        # from an intersection point should be negative. Here we use smallest distance
        # to see if it is a negative value, and use sigmoid for differentiability.
        
        inter_dist = -inter_dist
        min_inter_dist = inter_dist.min(dim=-1)[0]
        is_triangle = th.sigmoid(min_inter_dist * 1e3)
        
        return is_triangle

    def compute_half_plane_intersections(self, 
                                        half_planes: th.Tensor,
                                        half_plane_pair_indices: th.Tensor,):
        
        # compute the intersection point between a pair of half planes;

        half_planes_0 = \
                th.gather(half_planes, dim=1, 
                    index=half_plane_pair_indices[:, 0].unsqueeze(0).unsqueeze(-1).expand([half_planes.shape[0], -1, half_planes.shape[2]]))

        half_planes_1 = \
                th.gather(half_planes, dim=1, 
                    index=half_plane_pair_indices[:, 1].unsqueeze(0).unsqueeze(-1).expand([half_planes.shape[0], -1, half_planes.shape[2]]))

        inter0 = th.cross(half_planes_0, half_planes_1, dim=-1)

        # normalize [inter0] by dividing it with last dimension to get valid coordinates;
        # if last dimension was too small, the half planes are parallel and do not intersect;
        
        mask = th.abs(inter0[:, :, 2]) < self.EPS
        last_dim = th.where(mask,
                            th.ones_like(inter0[:, :, 2]),
                            inter0[:, :, 2])
        inter1 = inter0 / last_dim.unsqueeze(-1)
        
        # if there was no intersection, last dim is 1e7, not 1;

        inter = th.where(mask.unsqueeze(-1).expand((-1, -1, 3)),
                        th.ones_like(inter1) * 1e7,
                        inter1)

        return inter

        
    def compute_weighted_middle_points(self,
                                        nn_coords: th.Tensor,
                                        nn_weights: th.Tensor,
                                        centers: th.Tensor,
                                        centers_weights: th.Tensor,):
        
        # compute middle points between [center] points and [nn_coords] using weights;
        
        num_neighbors = nn_coords.shape[1]
        
        # expanding center points coordinates to match number of neighbor;
        # nn_coords.shape = [# point, # neighbor, 2]
        # centers.shape = [# point, 2]
        
        e_centers = centers.unsqueeze(1).expand([-1, num_neighbors, -1])
        
        # expanding center points weights to match number of neighbor;
        # nn_weights.shape = [# point, # neighbor]
        # center_weights.shape = [# point]
        
        e_centers_weights = centers_weights.unsqueeze(-1).expand([-1, num_neighbors])
        
        # square of offset between center points and neighbors;
        
        offset_square = th.square(e_centers - nn_coords)                        
        
        # [alpha] is the relative weight of center point w.r.t. neighbors in computing middle points; 
        # if alpha is large, middle point is close to the center point;
        # alpha.shape = [# point, # neighbor]
        
        alpha = th.where(th.sum(offset_square, dim=-1) < 1e-8,
                        
                        # if two neighboring points are too close, alpha becomes 0.5;
                        th.ones_like(nn_weights) * 0.5,
                        
                        # take into account distance between center and neighboring points;
                        # if they are far away, alpha becomes 0.5;
                        # if they are close, alpha decreases if the weight of the center point is larger;
                        # else, alpha increases if the weight of the center point is smaller;
                        
                        0.5 - ((e_centers_weights - nn_weights) / (2. * (th.sum(offset_square, dim=-1)))))
                       
        # shape to [# point, # neighbor, 2] 
        
        alpha = alpha.unsqueeze(-1).expand([-1, -1, 2])
        middle_points = e_centers * alpha + nn_coords * (1. - alpha)
        
        return middle_points

        
    def compute_half_planes(self, 
                            nn_coords: th.Tensor, 
                            nn_weights: th.Tensor, 
                            centers: th.Tensor,
                            centers_weights: th.Tensor):
        
        # compute equations for half planes between [centers] and [nn_coords],
        # taking into account their weights;
        
        num_points = nn_coords.shape[0]
        num_neighbors = nn_coords.shape[1]
        
        middle_points = self.compute_weighted_middle_points(nn_coords, nn_weights, centers, centers_weights)
        
        dir_vec = nn_coords - centers.unsqueeze(1)
        # dir_vec = gradient_clipping(dir_vec)
        
        # normalize [dir_vec];
        
        dir_vec_norm = th.norm(dir_vec, p=2, dim=-1, keepdim=True)
        dir_vec_norm = th.clamp(dir_vec_norm, min=1e-4)
        half_planes_normal = dir_vec / dir_vec_norm
        
        # project [middle_points] on [half_planes_normal];
        
        proj = -(middle_points[:, :, 0] * half_planes_normal[:, :, 0] + \
                    middle_points[:, :, 1] * half_planes_normal[:, :, 1])
        
        half_planes = th.cat([half_planes_normal, proj.unsqueeze(-1)], dim=-1)
        
        return half_planes
        
    def get_nn_size(self, num_points):
        
        return num_points - 1 if self.nn_size == -1 else self.nn_size
    
    def index_pairs(self, num_index, device):
        pairs = []
        
        for i in range(1, num_index):
            for j in range(i):
                pairs.append([i,j])
        
        pairs = th.tensor(pairs, dtype=th.int64, device=device)
        return pairs