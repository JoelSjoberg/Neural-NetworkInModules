import cupy as cp
def k_medians(clusters, points, iters = 10, method = cp.median):
    
    for i in range(iters):
        
        cluster_points = get_cluster_belonging(clusters, points)

        for c in range(len(clusters)):

            clusters[c] = method(cluster_points[c], axis = 0)
    
    return clusters

def get_cluster_belonging(clusters, points):
    cluster_points = [[] for i in range(len(clusters))]
    for e in points:

        minimum_distance = None

        cluster_dists = []

        for cluster in clusters:

            dist = cp.sqrt(cp.sum(cp.power(cluster - e, 2)))
            cluster_dists.append(dist)
        
        index = cp.argmin(cluster_dists)

        cluster_points[index].append(e)
        
    return cluster_points
