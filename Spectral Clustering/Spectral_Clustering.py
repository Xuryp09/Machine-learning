import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from PIL import Image
from Kernel_Kmeans import mult_RBFkernel

def compute_L_D(W):
    D = np.zeros((10000, 10000))
    L = np.zeros((10000, 10000))
    D = np.diag(np.sum(W, axis=1)) 
    L = D - W
    return L, D


def compute_L_sym(L, D):
    for i in range(10000):
        D[i][i] = 1.0 / np.sqrt(D[i][i])
    L_sym = D @ L @ D
    return L_sym

def compute_U(L, num_cluster):
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    sort_idx = np.argsort(eigenvalues)
    U = eigenvectors[:, sort_idx[1: num_cluster+1]]
    return U

def compute_T(U):  
    count = np.sum(U, axis = 1)
    for i in range(10000):
        U[i, :] /= count[i]
    return U
 
def choose_centroids(mode, A, num_cluster):
    if (mode == 0):
        return A[np.random.choice(10000, num_cluster)]
    else:
        y = np.zeros((num_cluster, num_cluster))
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
        
        for i in range(num_cluster):
           y[i] = A[centroids[i], :]
        
        return y

def kmeans(centroids, num_cluster, A):
    iteration = 100
    total = []
    clusters = np.zeros(10000)
    for l in range(iteration):
        new_clusters = np.zeros(10000)
        for i in range(10000):
            dist = np.zeros(num_cluster)
            for k in range(num_cluster):
                dist[k] = np.linalg.norm(A[i] - centroids[k])
            new_clusters[i] = np.argmin(dist)

        for k in range(num_cluster):
            centroids[k] = np.average(A[new_clusters == k], axis = 0)

        total.append(new_clusters)
        if(l == 0):
            visualization(A, num_cluster, new_clusters)

        if(np.linalg.norm(new_clusters - clusters) < 1e-2):
            break
        
        clusters = new_clusters
    return total

def make_gif(all_iteration_cluster, image_num, mode, cut, num_cluster):
    colors = np.array([[255,0,0],[0,255,0],[0,0,255],[95,0,135]])
    img = []
    for i in range(len(all_iteration_cluster)):
        temp = np.zeros((10000, 3))
        for j in range(10000):
            temp[j, :] = colors[int(all_iteration_cluster[i][j]), :]
    
        temp = temp.reshape((100, 100, 3))
        img.append(Image.fromarray(np.uint8(temp)))
    
    filename = f'./spectral_clustering/spectral_clustering_{image_num}_' \
               f'cluster{num_cluster}_' \
               f'{"normalized_cut" if cut else "ratio_cut"}_' \
               f'{"kmeans++" if mode else "random"}.gif'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    img[0].save(filename, save_all=True, append_images=img[1:], optimize=False, loop=0, duration=100)

def visualization(U, num_cluster, cluster):
    if(num_cluster == 2):
        plt.figure()
        x = U[:, 0]
        y = U[:, 1]
        plt.xlabel("1st dimension")
        plt.ylabel("2nd dimension")
        for i in range(num_cluster):
            plt.scatter(x[cluster == i], y[cluster == i], marker='.')
    elif(num_cluster == 3):
        figure = plt.figure()
        third = figure.gca(projection="3d")
        x = U[:, 0]
        y = U[:, 1]
        z = U[:, 2]
        third.set_xlabel("1st dimension")
        third.set_ylabel("2nd dimension")
        third.set_zlabel("3st dimension")
        for i in range(num_cluster):
            third.scatter(x[cluster == i], y[cluster == i], z[cluster == i], '.')
    plt.show()
    
if __name__ == '__main__':

    gamma_c = 1e-3
    gamma_s = 1e-3
    for image_num in range(1, 3):
        if(image_num == 1):
            image = Image.open('image1.png')
        else:
            image = Image.open('image2.png')
        image_array = np.array(image).reshape(10000, 3)
        W = mult_RBFkernel(image_array, gamma_c, gamma_s)
        L, D = compute_L_D(W)
        for mode in range(2):
            for num_cluster in range(2, 5):
                for cut in range(2):
                        if(cut == 0):
                            U_ratio = compute_U(L, num_cluster)
    
                        if(cut == 1):
                            L_sym = compute_L_sym(L, D)
                            U_normalize = compute_U(L_sym, num_cluster)
                            T = compute_T(U_normalize)

                        print("clustering...")
                        if(cut == 0):
                            centroids = choose_centroids(mode, U_ratio, num_cluster)
                            all_iteration_cluster = kmeans(centroids, num_cluster, U_ratio)
                        else:
                            centroids = choose_centroids(mode, T, num_cluster)
                            all_iteration_cluster = kmeans(centroids, num_cluster, T)
    
                        print("making gif...")
                        make_gif(all_iteration_cluster, image_num, mode, cut, num_cluster)
    