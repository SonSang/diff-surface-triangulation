import numpy as np
import torch as th
import time
import os

from torch_impl.wdt2d_s1_ours import WDT as WDT_OURS
from torch_impl.wdt2d_s1_prev import WDT as WDT_PREV
from torch_impl.wdt2d_ours import tensor_intersect, tensor_subtract

device = 'cuda:0'
n_points = 100

th.random.manual_seed(1)

save_path =  f"result/th/wdt/precision/ours/s1/{int(np.floor(time.time()))}"

if not os.path.exists(save_path):
    os.makedirs(save_path)

# set of random 2d points that we are going to compute WDT for;
points = th.rand((n_points, 2), dtype=th.float32, device=device)

# randomly initialize weights for each point;
weights = th.randn((n_points), dtype=th.float32, device=device) * 1e-3

wdt_ours = WDT_OURS(device)
wdt_prev = WDT_PREV(device)

'''
Collect query edges that will be fed into algorithm
'''
tris = wdt_ours.nond_wdt(points, (weights).unsqueeze(-1))  # existing faces
edges = [tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]] # existing edges
edges = th.cat(edges, dim=0)
edges = th.sort(edges, dim=-1)[0]
edges = th.unique(edges, dim=0)
e_query_edges = edges

# KNN-based update (every point)
query_k = 20
every_point_knn_idx = wdt_ours.find_knn(points, points, query_k)
every_point_knn_idx = every_point_knn_idx[:, 1:]   # exclude self

# [# points, # combs, 1]
query_edges = every_point_knn_idx.unsqueeze(-1)

# [# points, # combs, 1]
point_idx = th.arange(len(points), device=device)
point_idx = point_idx.unsqueeze(-1).expand([-1, len(query_edges[0])])
point_idx = point_idx.unsqueeze(-1)

# [# points, # combs, 2]
query_edges = th.cat([point_idx, query_edges], dim=-1)
query_edges = th.reshape(query_edges, [-1, 2])
query_edges = th.sort(query_edges, dim=-1)[0]
ne_query_edges = th.unique(query_edges, dim=0)
ne_query_edges = tensor_subtract(ne_query_edges, e_query_edges)
if len(ne_query_edges) > len(e_query_edges):
    randperm = th.randperm(len(ne_query_edges), device=device)
    ne_query_edges = ne_query_edges[randperm[:len(e_query_edges)]]

query_edges = th.cat([e_query_edges, ne_query_edges], dim=0)
query_edges = th.sort(query_edges, dim=-1)[0]
query_edges = th.unique(query_edges, dim=0)

print("=== Num. of query edges: ", len(query_edges))

'''
Run algorithm
'''
# ours
start_time = time.time()
with th.no_grad():
    our_edges, our_probs = wdt_ours.forward(points, weights, query_edges)
end_time = time.time()
print(f"(Ours) Elapsed time: {end_time - start_time} sec")


'''
Compute precision: Ours
'''

print("=== Ours ===")

# false positive: ratio of faces that are determined
# as exist, but in fact does not exist
positive_faces = our_edges[our_probs > 0.5] # dt_indices[is_dt_discrete.to(dtype=th.bool)]
real_positive_faces = tensor_intersect(e_query_edges, positive_faces)
num_positive_faces = positive_faces.shape[0]
num_real_positive_faces = real_positive_faces.shape[0]
true_positive_ratio = num_real_positive_faces / num_positive_faces
false_positive_ratio = 1 - true_positive_ratio
print(f"True positive ratio: {true_positive_ratio}")
print(f"False positive ratio: {false_positive_ratio}")

# false negative: ratio of faces that are determined
# as not exist, but in fact exist
negative_faces = our_edges[our_probs < 0.5]
real_negative_faces = tensor_intersect(ne_query_edges, negative_faces)
num_negative_faces = negative_faces.shape[0]
num_real_negative_faces = real_negative_faces.shape[0]
true_negative_ratio = num_real_negative_faces / num_negative_faces
false_negative_ratio = 1 - true_negative_ratio
print(f"True negative ratio: {true_negative_ratio}")
print(f"False negative ratio: {false_negative_ratio}")

# completeness: ratio of real faces that are determined
# as exist by dWDT
# real_positive_faces = tensor_intersect(real_tris, dt_indices)
# num_real_faces = real_tris.shape[0]
# num_real_positive_faces = real_positive_faces.shape[0]
# completeness = num_real_positive_faces / num_real_faces
# print(f"Completeness: {completeness}")


'''
Compute precision: Prev
'''

# prev
try:
    start_time = time.time()
    with th.no_grad():
        prev_edges, prev_probs = wdt_prev.forward(points, weights, query_edges)
    end_time = time.time()
    print(f"(Prev) Elapsed time: {end_time - start_time} sec")

    print("=== Prev ===")

    # false positive: ratio of faces that are determined
    # as exist, but in fact does not exist
    positive_faces = prev_edges[prev_probs > 0.5] # dt_indices[is_dt_discrete.to(dtype=th.bool)]
    real_positive_faces = tensor_intersect(e_query_edges, positive_faces)
    num_positive_faces = positive_faces.shape[0]
    if num_positive_faces == 0:
        true_positive_ratio = 0.0
        false_positive_ratio = 1.0
    else:
        num_real_positive_faces = real_positive_faces.shape[0]
        true_positive_ratio = num_real_positive_faces / num_positive_faces
        false_positive_ratio = 1 - true_positive_ratio
    print(f"True positive ratio: {true_positive_ratio}")
    print(f"False positive ratio: {false_positive_ratio}")

    # false negative: ratio of faces that are determined
    # as not exist, but in fact exist
    negative_faces = prev_edges[prev_probs < 0.5]
    real_negative_faces = tensor_intersect(ne_query_edges, negative_faces)
    num_negative_faces = negative_faces.shape[0]
    if num_negative_faces == 0:
        true_negative_ratio = 0.0
        false_negative_ratio = 1.0
    else:
        num_real_negative_faces = real_negative_faces.shape[0]
        true_negative_ratio = num_real_negative_faces / num_negative_faces
        false_negative_ratio = 1 - true_negative_ratio
    print(f"True negative ratio: {true_negative_ratio}")
    print(f"False negative ratio: {false_negative_ratio}")
except:
    print("Prev failed")