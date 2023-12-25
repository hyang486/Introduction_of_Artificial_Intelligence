def hac(features):
    n = len(features)  # Number of initial clusters
    
    # Step 1: Create a distance matrix
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            distance_matrix[i, j] = np.linalg.norm(features[i] - features[j])
            distance_matrix[j, i] = distance_matrix[i, j]  # Symmetric matrix
    
    # Step 2: Initialize clusters and the result array
    clusters = [{i} for i in range(n)]  # Each point in its own cluster
    Z = np.zeros((n-1, 4))
    
    # Step 3: Agglomerative clustering
    new_cluster_idx = n
    for i in range(n-1):
        # a) Find the pair of clusters that are closest
        min_dist = np.inf
        min_pair = None
        for j in range(len(clusters)):
            for k in range(j+1, len(clusters)):
                dist = max(distance_matrix[x, y] for x in clusters[j] for y in clusters[k])
                if dist < min_dist or (dist == min_dist and (min(j,k) < min(min_pair))):  # Tie-breaking
                    min_dist = dist
                    min_pair = (j, k)
        
        # b) Merge them and update the distance matrix
        merged_cluster = clusters[min_pair[0]].union(clusters[min_pair[1]])
        clusters.append(merged_cluster)
        
        # Update Z
        Z[i, 0], Z[i, 1] = min_pair
        Z[i, 2] = min_dist
        Z[i, 3] = len(merged_cluster)
        
        # Remove old clusters
        clusters.pop(max(min_pair))
        clusters.pop(min(min_pair))

        new_cluster_idx += 1
               
    return Z