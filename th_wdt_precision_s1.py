import numpy as np
import torch as th
import time
import os

from torch_impl.wdt2d_s1 import WDT1 as WDT
from torch_impl.wdt2d_ours import WDT as WDT_OURS
from torch_impl.wdt2d_ours import tensor_intersect

device = 'cuda:0'
n_points = 1000

# 1. used to build ''virtual'' Voronoi cell
# 2. used to find query faces
N_NEAREST_NEIGHBORS = 100 # n_points - 1

# used to find query faces
# this maximum value lets us to consider every
# combination of 2 nearest neighbors
n_trigs = (N_NEAREST_NEIGHBORS * (N_NEAREST_NEIGHBORS - 1)) * 2       # max = Comb(N_NEAREST_NEIGHBORS, 2) * 4

th.random.manual_seed(1)

save_path =  f"result/th/wdt/precision/s1/{int(np.floor(time.time()))}"

if not os.path.exists(save_path):
    os.makedirs(save_path)

# set of random 2d points that we are going to compute WDT for;
points = th.rand((n_points, 2), dtype=th.float32, device=device)

# randomly initialize weights for each point;
weights = th.randn((n_points), dtype=th.float32, device=device) * 1e-3

wdt = WDT(n_trigs, nn_size=N_NEAREST_NEIGHBORS)
wdt2 = WDT_OURS(device)

# approximation
start_time = time.time()
with th.no_grad():
    is_dt_discrete, is_dt_smooth, dt_indices = wdt(points, weights)
end_time = time.time()

print(f"Elapsed time: {end_time - start_time} sec")

dt_indices = th.sort(dt_indices, dim=1)[0]
# dt_indices = th.unique(dt_indices, dim=0)

'''
Compute precision
'''
start_time = time.time()
real_tris = wdt2.nond_wdt(points, (weights).unsqueeze(-1))
end_time = time.time()
print(f"Elapsed time: {end_time - start_time} sec")

real_edges = []
pairs = [[0, 1], [1, 2], [2, 0]]
for pair in pairs:
    real_edges.append(real_tris[:, pair])
real_edges = th.cat(real_edges, dim=0)
real_edges = th.sort(real_edges, dim=1)[0]
real_edges = th.unique(real_edges, dim=0)
real_tris = real_edges

# false positive: ratio of faces that are determined
# as exist, but in fact does not exist
positive_faces = dt_indices[is_dt_discrete.to(dtype=th.bool)]
real_positive_faces = tensor_intersect(real_tris, positive_faces)
num_positive_faces = positive_faces.shape[0]
num_real_positive_faces = real_positive_faces.shape[0]
true_positive_ratio = num_real_positive_faces / num_positive_faces
false_positive_ratio = 1 - true_positive_ratio
print(f"True positive ratio: {true_positive_ratio}")
print(f"False positive ratio: {false_positive_ratio}")

# false negative: ratio of faces that are determined
# as not exist, but in fact exist
negative_faces = dt_indices[~is_dt_discrete.to(dtype=th.bool)]
real_negative_faces = tensor_intersect(real_tris, negative_faces)
num_negative_faces = negative_faces.shape[0]
num_real_negative_faces = real_negative_faces.shape[0]
false_negative_ratio = num_real_negative_faces / num_negative_faces
true_negative_ratio = 1 - false_negative_ratio
print(f"True negative ratio: {true_negative_ratio}")
print(f"False negative ratio: {false_negative_ratio}")

# completeness: ratio of real faces that are determined
# as exist by dWDT
real_positive_faces = tensor_intersect(real_tris, dt_indices)
num_real_faces = real_tris.shape[0]
num_real_positive_faces = real_positive_faces.shape[0]
completeness = num_real_positive_faces / num_real_faces
print(f"Completeness: {completeness}")