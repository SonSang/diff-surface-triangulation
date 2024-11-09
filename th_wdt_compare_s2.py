import numpy as np
import torch as th
import time
import os

from torch_impl.wdt2d_ours import WDT as WDT_OURS
from torch_impl.wdt2d_prev import WDT as WDT_PREV
from torch_impl.wdt2d_ours import tensor_intersect, tensor_subtract

device = 'cuda:0'
n_points = 1000000

th.random.manual_seed(1)

save_path =  f"result/th/wdt/precision/ours/s2/{int(np.floor(time.time()))}"

if not os.path.exists(save_path):
    os.makedirs(save_path)

# set of random 2d points that we are going to compute WDT for;
points = th.rand((n_points, 2), dtype=th.float32, device=device)

# randomly initialize weights for each point;
weights = th.randn((n_points), dtype=th.float32, device=device) * 1e-3

wdt_ours = WDT_OURS(device)
wdt_prev = WDT_PREV(device)

'''
Collect query faces that will be fed into algorithm
'''
e_query_tris = wdt_ours.nond_wdt(points, (weights).unsqueeze(-1))  # existing faces

# KNN-based update (every point)
query_k = 20
every_point_knn_idx = wdt_ours.find_knn(points, points, query_k)
every_point_knn_idx = every_point_knn_idx[:, 1:]   # exclude self

# [# combs, 2]
combs = th.combinations(th.arange(query_k - 1, device=device), 2)
# [# points, # combs, 2]
combs = combs.unsqueeze(0).expand([len(points), -1, -1])
# [# points, # combs, # k - 1]
every_point_knn_idx = every_point_knn_idx.unsqueeze(1).expand([-1, len(combs[0]), -1])

# [# points, # combs, 2]
query_faces = th.gather(every_point_knn_idx, dim=-1, index=combs)

# [# points, # combs, 1]
point_idx = th.arange(len(points), device=device)
point_idx = point_idx.unsqueeze(-1).expand([-1, len(query_faces[0])])
point_idx = point_idx.unsqueeze(-1)

# [# points, # combs, 3]
query_faces = th.cat([point_idx, query_faces], dim=-1)
query_faces = th.reshape(query_faces, [-1, 3])
query_faces = th.sort(query_faces, dim=-1)[0]
ne_query_faces = th.unique(query_faces, dim=0)
ne_query_faces = tensor_subtract(ne_query_faces, e_query_tris)
if len(ne_query_faces) > len(e_query_tris):
    randperm = th.randperm(len(ne_query_faces), device=device)
    ne_query_faces = ne_query_faces[randperm[:len(e_query_tris)]]

query_faces = th.cat([e_query_tris, ne_query_faces], dim=0)
query_faces = th.sort(query_faces, dim=-1)[0]
query_faces = th.unique(query_faces, dim=0)

print("=== Num. of query faces: ", len(query_faces))

'''
Run algorithm
'''
# ours
start_time = time.time()
with th.no_grad():
    our_faces, our_probs = wdt_ours.forward(points, weights, query_faces)
end_time = time.time()
print(f"(Ours) Elapsed time: {end_time - start_time} sec")

# prev
prev_failed = False
try:
    start_time = time.time()
    with th.no_grad():
        prev_faces, prev_probs = wdt_prev.forward(points, weights, query_faces)
    end_time = time.time()
    print(f"(Prev) Elapsed time: {end_time - start_time} sec")
except:
    print("Prev failed")
    prev_failed = True

'''
Compute precision: Ours
'''

print("=== Ours ===")

# false positive: ratio of faces that are determined
# as exist, but in fact does not exist
positive_faces = our_faces[our_probs > 0.5] # dt_indices[is_dt_discrete.to(dtype=th.bool)]
real_positive_faces = tensor_intersect(e_query_tris, positive_faces)
num_positive_faces = positive_faces.shape[0]
num_real_positive_faces = real_positive_faces.shape[0]
true_positive_ratio = num_real_positive_faces / num_positive_faces
false_positive_ratio = 1 - true_positive_ratio
print(f"True positive ratio: {true_positive_ratio}")
print(f"False positive ratio: {false_positive_ratio}")

# false negative: ratio of faces that are determined
# as not exist, but in fact exist
negative_faces = our_faces[our_probs < 0.5]
real_negative_faces = tensor_intersect(ne_query_faces, negative_faces)
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

if prev_failed:
    exit()

print("=== Prev ===")

# false positive: ratio of faces that are determined
# as exist, but in fact does not exist
positive_faces = prev_faces[prev_probs > 0.5] # dt_indices[is_dt_discrete.to(dtype=th.bool)]
real_positive_faces = tensor_intersect(e_query_tris, positive_faces)
num_positive_faces = positive_faces.shape[0]
num_real_positive_faces = real_positive_faces.shape[0]
true_positive_ratio = num_real_positive_faces / num_positive_faces
false_positive_ratio = 1 - true_positive_ratio
print(f"True positive ratio: {true_positive_ratio}")
print(f"False positive ratio: {false_positive_ratio}")

# false negative: ratio of faces that are determined
# as not exist, but in fact exist
negative_faces = prev_faces[prev_probs < 0.5]
real_negative_faces = tensor_intersect(ne_query_faces, negative_faces)
num_negative_faces = negative_faces.shape[0]
num_real_negative_faces = real_negative_faces.shape[0]
true_negative_ratio = num_real_negative_faces / num_negative_faces
false_negative_ratio = 1 - true_negative_ratio
print(f"True negative ratio: {true_negative_ratio}")
print(f"False negative ratio: {false_negative_ratio}")