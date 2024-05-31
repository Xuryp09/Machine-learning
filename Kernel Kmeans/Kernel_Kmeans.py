import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from PIL import Image
import os

def mult_RBFkernel(image, gamma_c, gamma_s):
    spatial = np.zeros((10000, 2))
    for i in range(100):
        for j in range(100):
            spatial[i * 100 + j][0] = i
            spatial[i * 100 + j][1] = j

    spatial_dist = cdist(spatial, spatial, 'sqeuclidean')
    spatial_RBF = np.exp(-gamma_s * spatial_dist)

    color_dist = cdist(image, image, 'sqeuclidean')
    color_RBF = np.exp(-gamma_c * color_dist)

    return np.multiply(spatial_RBF ,color_RBF)

def choose_centroids(mode, num_cluster):
    if(mode == 0):
        centroids = np.random.choice(10000, num_cluster)
    else:
        centroids = np.zeros(num_cluster)
        centroids[0] =  np.random.choice(10000, 1)
        for k in range(1, num_cluster):
            dist = np.zeros(10000)
            for i in range(10000):
                shortest = np.inf
                for j in range(k):
                    temp_x = i // 100
                    temp_y = i % 100
                    temp_dist = np.array([temp_x - (centroids[j] // 100)], temp_y - (centroids[j] % 100))
                    temp = np.linalg.norm(temp_dist)
                    if (temp < shortest):
                        shortest = temp
                dist[i] = shortest
            dist /= np.sum(dist)
            centroids[k] = np.random.choice(10000, 1, p = dist)[0]
        centroids = centroids.astype(int)
    return centroids

def initial_kmeans_clustering(centroids, num_cluster, new_kernel):
    clusters = np.zeros(10000)
    for i in range(10000):
        dist = np.zeros(num_cluster)
        for k in range(num_cluster):
            dist[k] = new_kernel[i, i] - 2 * new_kernel[i, centroids[k]] + new_kernel[centroids[k], centroids[k]]
        clusters[i] = np.argmin(dist)
    return clusters

def kernel_kmeans(clusters, new_kernel):
    iteration = 100
    total = []
    total.append(clusters)
    for _ in range(iteration):
        new_clusters = np.zeros(10000)
        C_k = np.zeros(num_cluster)
        second_term = np.zeros((10000, num_cluster))
        third_term = np.zeros(num_cluster)
        for k in range(num_cluster):
            C_k[k] = np.sum(clusters == k)

        for k in range(num_cluster):
            mask = clusters == k
            second_term[:, k] = np.sum(new_kernel[:, mask], axis = 1) * 2 / C_k[k]

        for k in range(num_cluster):
            mask = clusters == k
            third_term[k] = np.sum(new_kernel[np.ix_(mask, mask)]) / np.power(C_k[k], 2)
        
        for i in range(10000):
            dist = np.zeros(num_cluster)
            for k in range(num_cluster):
                dist[k] = new_kernel[i][i] - second_term[i][k] + third_term[k]
            new_clusters[i] = np.argmin(dist)

        total.append(new_clusters)
        if(np.linalg.norm(new_clusters - clusters) < 1e-2):
            break
        
        clusters = new_clusters
    
    return total

def make_gif(all_iteration_cluster, image_num, mode, num_cluster):
    colors = np.array([[255,0,0],[0,255,0],[0,0,255],[95,0,135]])
    img = []
    for i in range(len(all_iteration_cluster)):
        temp = np.zeros((10000, 3))
        for j in range(10000):
            temp[j, :] = colors[int(all_iteration_cluster[i][j]), :]
    
        temp = temp.reshape((100, 100, 3))
        img.append(Image.fromarray(np.uint8(temp)))
    
    filename = f'./kernel_kmeans/kernel_kmeans_{image_num}_' \
               f'cluster{num_cluster}_' \
               f'{"kmeans++" if mode else "random"}.gif'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    img[0].save(filename, save_all=True, append_images=img[1:], optimize=False, loop=0, duration=100)

if __name__ == '__main__':

    gamma_c = 1e-3
    gamma_s = 1e-3

    for image_num in range(1, 2):
        for mode in range(1, 2):
            for num_cluster in range(2, 3):
                if(image_num == 1):
                    image = Image.open('image1.png')
                else:
                    image = Image.open('image2.png')
                image_array = np.array(image).reshape(10000, 3)

                print("calculate kernel...")
                new_kernel = mult_RBFkernel(image_array, gamma_c, gamma_s)
                print("find the centroids...")
                centroids = choose_centroids(mode, num_cluster)
                print("compute kmeans...")
                clusters = initial_kmeans_clustering(centroids, num_cluster, new_kernel)
                all_iteration_cluster = kernel_kmeans(clusters, new_kernel)
                print("making gif...")
                make_gif(all_iteration_cluster, image_num, mode, num_cluster) 