import numpy as np

class SimpleDecoder:
    """Linear classifier for evaluation."""
    def __init__(self):
        self.W = None
        
    def fit(self, X, y):
        n_classes = int(y.max()) + 1  # Convert to int
        Y = np.zeros((len(y), n_classes))
        Y[np.arange(len(y)), y.astype(int)] = 1  # Ensure indices are integers
        self.W = np.linalg.pinv(X) @ Y
        
    def predict(self, X):
        return np.argmax(X @ self.W, axis=1)
    
    def score(self, X, y):
        return np.mean(self.predict(X) == y)