import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
import numpy as np
DEBUG = False # sets result saving
import tensorflow as tf
sys.path.append(os.path.join(BASE_DIR, 'models'))
import optimization_utils as utils
import model
import time
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

n_points = 10
N_NEAREST_NEIGHBORS = n_points - 1
n_trigs = (N_NEAREST_NEIGHBORS * (N_NEAREST_NEIGHBORS - 1)) * 2       # max = Comb(N_NEAREST_NEIGHBORS, 2) * 4

def init_graph(point_set, weights, tpoint_set):
    config = utils.init_config()
    
    with tf.device('/gpu:'+str(0)):
        # learning_rate = tf.compat.v1.placeholder(tf.float32, shape=[])
        
        point_set = tf.Variable(point_set, dtype=tf.float32)
        weights = tf.Variable(weights, dtype=tf.float32)
        tpoint_set = tf.constant(tpoint_set, dtype=tf.float32)
        
        use_weights = tf.minimum(tf.abs(weights), 0.002)
        neighbors = utils.compute_nearest_neighbors(point_set, N_NEAREST_NEIGHBORS)
        
        neighbor_points = tf.gather(point_set, neighbors)
        neighbor_points = neighbor_points - neighbor_points[:,0:1]
        neighbor_points = tf.concat([neighbor_points, tf.zeros([BATCH_SIZE,  N_NEAREST_NEIGHBORS+1, 1])], axis = 2)
        neighbor_weights = tf.gather(use_weights, neighbors)
        
        normals = np.zeros([BATCH_SIZE, 3])
        normals[:, 2] = 1
        normals = tf.constant(normals, dtype=tf.float32)
        points_idx = tf.constant(np.array(range(N_NEAREST_NEIGHBORS+1)))
        points_idx  = tf.tile(points_idx[tf.newaxis, :], [BATCH_SIZE, 1])
        
        # @sanghyun: Do WDT;
        # [target_approx_triangles]: smooth inclusion score for each triangle;
        # [target_indices]: vertex indices of each triangle;
        # [exact_triangles]: hard inclusion score for each triangle;
        # [local_indices]: local vertex indices that make up triangle; (local means for each center points)
        target_approx_triangles, target_indices, _, exact_triangles, local_indices = model.get_triangles_geo_batches(neighbor_points,
                                                                                                                        normals,
                                                                                                                        tf.abs(neighbor_weights),
                                                                                                                        n_neighbors=N_NEAREST_NEIGHBORS,
                                                                                                                        n_trigs=n_trigs,
                                                                                                                        gdist=neighbor_points,
                                                                                                                        gdist_neighbors=neighbors[:, 1:], # points_idx[:,1:],
                                                                                                                        first_index=tf.range(0, n_points, dtype=tf.int32)) # tf.zeros([BATCH_SIZE], dtype=tf.int64))
        
        # 
        
        init = tf.compat.v1.global_variables_initializer()
    
    session = tf.compat.v1.Session(config=config)
    session.run(init)

    ops = {"triangles":target_approx_triangles,
            "indices":target_indices,
            "use_weights": use_weights,
            "neighbors": neighbors,
            "neighbor_points": neighbor_points,
            "weights": weights,
            "points": point_set,
            "exact_triangles":exact_triangles,
            }

    return session, ops


def run_graph(session, ops, save_path):
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    to_run = [ops['triangles'], ops['indices'], ops['points'], ops['weights'], ops['neighbors'], ops['exact_triangles']]
    triangles, indices, points, weights, neighbors, exact_triangles = session.run(to_run, feed_dict=None)
    
    # render results;
    
    # points with weights;
    
    plt.figure()
    wstd = np.sqrt(np.var(weights))
    wstd = max(wstd, 1e-4)
    n_weights = (weights - np.mean(weights)) / wstd
    n_weights = n_weights - np.min(n_weights) + 0.01
    n_weights = n_weights * 10.
    
    plt.scatter(points[:, 0], points[:, 1], s=n_weights, alpha=1.0)
    plt.savefig(f"{save_path}/weighted_points.png")
    
    # soft triangles;
    
    # hard triangles;
    
    hard_triangle_indices = np.where(exact_triangles > 0.5)
    hard_triangle_indices = indices[hard_triangle_indices]
    
    plt.figure()
    plt.scatter(points[:, 0], points[:, 1], s=n_weights, alpha=0.5)
    for i, hti in enumerate(hard_triangle_indices):
        point0, point1 = points[hti[0]], points[hti[1]]
        x, y = [point0[0], point1[0]], [point0[1], point1[1]]
        plt.plot(x, y, color='r', linestyle='-', linewidth=1)
        
        point0, point1 = points[hti[1]], points[hti[2]]
        x, y = [point0[0], point1[0]], [point0[1], point1[1]]
        plt.plot(x, y, color='r', linestyle='-', linewidth=1)
        
        point0, point1 = points[hti[0]], points[hti[2]]
        x, y = [point0[0], point1[0]], [point0[1], point1[1]]
        plt.plot(x, y, color='r', linestyle='-', linewidth=1)
    plt.plot(points[:,0], points[:,1], 'o')
        
    plt.savefig(f"{save_path}/hard_triangles.png")
    
    # for comparison to original DT;
    tri = Delaunay(points)
    
    plt.figure()
    plt.triplot(points[:,0], points[:,1], tri.simplices)
    plt.plot(points[:,0], points[:,1], 'o')
    
    plt.savefig(f"{save_path}/vanilla_dt_triangles.png")

if __name__ == '__main__':
    # tf.compat.v1.disable_eager_execution()
    
    np.random.seed(2)
    
    save_path =  f"result/wdt/{np.floor(time.time())}"

    # set of random 2d points that we are going to use as a starting point;
    point_set = np.random.random((n_points, 2)) * 2.0 - 1.0
    
    # randomly initialize weights for each point;
    weights = np.random.uniform(size=n_points) * 0.00005 + 0.0001
    
    # set of points on a unit circle that we are going to approximate;
    num_target_point_set = 1000
    theta = np.arange(0., 1., 1./num_target_point_set)
    target_point_set = np.stack([np.cos(theta), np.sin(theta)], axis=-1)
    
    BATCH_SIZE = n_points
                
    session, ops = init_graph(point_set, weights, target_point_set)
    
    run_graph(session, ops, save_path)
