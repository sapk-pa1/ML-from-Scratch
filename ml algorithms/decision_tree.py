


import numpy as np 


def entropy(y): 
    """calculates the entropy of the feature 

    Args:
        y (vector): class labels

    Returns:
        float : entropy 
    """
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([p*np.log2(p) for p in ps if p>0]) # log is not defined for negative numbers 

class Node: 
    def __init__(self,feature = None , threshold = None , left = None , right = None, *, value = None  ): 
        self.feature = feature
        self.threshold = threshold
        self.left = left 
        self.right = right 
        self.value = value 
    def is_leaf_node(self): 
        return self.value is not None 
    
class DecisionTree: 
    def __init__(self, min_samples_split= 2 , max_depth = 100 , n_features = None ):
        self.min_samples_split = min_samples_split
        self.max_depth =  max_depth
        self.n_feats = n_features
        self.root = None 
    def fit(self, X, y): 
        # grow tree 
        self.n_features =  X.shape[1] if not self.n_feats else min(self.n_features, X.shape[1])
        self.root = self._grow_tree(X,y)
    
    def _grow_tree(self, X, y , depth = 0 ) : 
        n_samples , n_features = X.shape 
        n_labels = len(np.unique(y))

        #stopping criteria 
        if (depth >= self.max_depth
            or n_labels ==1 
            or n_samples < self.min_samples.split): 
            leaf_value = self.most_common_label(y)
            return Node(value = leaf_value)


    def predict(self, X): 
        # traverse tree 
        pass 
