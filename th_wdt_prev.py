import numpy as np
import torch as th
import time
import os

from torch_impl.wdt2d import WDT
import matplotlib.pyplot as plt

from tqdm import tqdm

device = 'cuda:0'
n_points = 50
N_NEAREST_NEIGHBORS = 20
n_trigs = (N_NEAREST_NEIGHBORS * (N_NEAREST_NEIGHBORS - 1)) * 2       # max = Comb(N_NEAREST_NEIGHBORS, 2) * 4

def optimize(initial_points: th.Tensor, 
            initial_weights: th.Tensor, 
            target_points: th.Tensor,
            save_path: str):
    
    points = initial_points.detach().clone()
    weights = initial_weights.detach().clone()
    
    points.requires_grad = True
    weights.requires_grad = True
    
    optimizer = th.optim.Adam([points, weights], 1e-3)
    
    wdt = WDT(n_trigs)
    
    bar = tqdm(range(1000))
    
    for iter in bar:
        
        is_dt_discrete, is_dt_smooth, dt_indices = wdt(points, weights)
        
        # render results;
    
        with th.no_grad():
            if iter % 100 == 0 or iter == 999:
                discrete_dt_indices = th.where(is_dt_discrete > 0.5)
                discrete_dt_indices = dt_indices[discrete_dt_indices]
                
                plt.figure()
                for hti in discrete_dt_indices:
                    point0, point1 = points[hti[0]].cpu().numpy(), points[hti[1]].cpu().numpy()
                    x, y = [point0[0], point1[0]], [point0[1], point1[1]]
                    plt.plot(x, y, color='r', linestyle='-', linewidth=1)
                    
                    point0, point1 = points[hti[1]].cpu().numpy(), points[hti[2]].cpu().numpy()
                    x, y = [point0[0], point1[0]], [point0[1], point1[1]]
                    plt.plot(x, y, color='r', linestyle='-', linewidth=1)
                    
                    point0, point1 = points[hti[0]].cpu().numpy(), points[hti[2]].cpu().numpy()
                    x, y = [point0[0], point1[0]], [point0[1], point1[1]]
                    plt.plot(x, y, color='r', linestyle='-', linewidth=1)
                
                #n_weights = (weights - weights.mean()) / th.clamp(th.sqrt(weights.var()), min=1e-4)
                # plt.scatter(points[:,0].cpu().numpy(),
                #             points[:,1].cpu().numpy(),
                #             s=weights.cpu().numpy() * 1e0,)
                plt.plot(points[:,0].cpu().numpy(), points[:,1].cpu().numpy(), 'o', markersize=1e-0)
                plt.plot(target_points[:,0].cpu().numpy(), target_points[:,1].cpu().numpy(), 'x', markersize=1e-0)
                plt.savefig(f"{save_path}/iteration_{iter}.png")
                plt.close()
    
        # compute loss;
        
        e_points = points.unsqueeze(0).expand([dt_indices.shape[0], -1, -1])
        e_dt_indices = dt_indices.unsqueeze(-1).expand([-1, -1, points.shape[-1]])
        dt_coords = th.gather(e_points, dim=1, index=e_dt_indices)
        
        dt_edge_beg = dt_coords.clone()
        dt_edge_end = dt_coords.clone()
        dt_edge_end[:, 0, :], dt_edge_end[:, 1, :], dt_edge_end[:, 2, :] = \
            dt_edge_end[:, 1, :], dt_edge_end[:, 2, :], dt_edge_end[:, 0, :]
        
        dt_edge_dir = dt_edge_end - dt_edge_beg
        
        e_target_points = target_points.unsqueeze(0).unsqueeze(1).expand([dt_indices.shape[0], 3, -1, -1])
        e_dt_edge_beg = dt_edge_beg.unsqueeze(2).expand([-1, -1, target_points.shape[0], -1])
        e_target_dir = e_target_points - e_dt_edge_beg
        
        e_dt_edge_dir = dt_edge_dir.unsqueeze(2).expand([-1, -1, target_points.shape[0], -1])
        
        target_proj = th.sum(e_target_dir * e_dt_edge_dir, dim=-1, keepdim=True)
        target_proj = th.clamp(target_proj, min=0., max=1.)
        target_proj_coords = e_dt_edge_beg + e_dt_edge_dir * target_proj
        
        target_dist = th.norm(e_target_points - target_proj_coords, p=2, dim=-1)
        target_dist = th.transpose(target_dist, 0, 2)
        target_dist = th.transpose(target_dist, 1, 2)
        target_dist = th.min(target_dist, dim=-1)[0]
        
        # clamp [is_dt_smooth] for back prop;
        is_dt_smooth = th.clamp(is_dt_smooth, min=1e-4)
        smooth_target_dist = target_dist / is_dt_smooth.unsqueeze(0)
        smooth_target_dist = th.min(smooth_target_dist, dim=-1)[0]
        
        loss = th.mean(smooth_target_dist) # + th.mean(weights) * 1e-2
        
        optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_([points, weights], 1.)
        optimizer.step()
        
        bar.set_description(f"Loss: {loss:.4f}")

if __name__ == '__main__':
    
    # th.random.manual_seed(1)
    np.random.seed(1)
    
    save_path =  f"result/th/wdt/prev/{int(np.floor(time.time()))}"
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # set of random 2d points that we are going to compute WDT for;
    # points = th.rand((n_points, 2), dtype=th.float32, device=device)
    points = th.tensor(np.random.random((n_points, 2)), dtype=th.float32, device=device)
    
    # randomly initialize weights for each point;
    # weights = th.rand((n_points), dtype=th.float32, device=device) * 0.005 + 0.01
    weights = th.tensor(np.random.uniform(size=n_points) * 0.005 + 0.01, dtype=th.float32, device=device)
    # weights[:] = 10.
    
    params = th.arange(100, device=device) / 1e2 * th.pi * 2.0
    target_points_x = 0.5 + 0.5 * th.cos(params)
    target_points_y = 0.5 + 0.5 * th.sin(params)
    target_points = th.stack([target_points_x, target_points_y], dim=-1)
    
    optimize(points, weights, target_points, save_path)