import numpy as np

def relief(X, y, k=3):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    
    for i in range(n_samples):
        instance = X[i]
        near_hit = None
        near_miss = None
        hit_dist = float('inf')
        miss_dist = float('inf')
        
        for j in range(n_samples):
            if y[j] == y[i]:
                dist = np.linalg.norm(X[j] - instance)
                if dist < hit_dist and dist != 0:
                    near_hit = X[j]
                    hit_dist = dist
            else:
                dist = np.linalg.norm(X[j] - instance)
                if dist < miss_dist:
                    near_miss = X[j]
                    miss_dist = dist
        
        weights += np.abs(instance - near_hit) - np.abs(instance - near_miss)
    
    return weights / n_samples

# Example usage
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
y = np.array([0, 1, 0])
feature_weights = relief(X, y)

print("Feature weights:", feature_weights)