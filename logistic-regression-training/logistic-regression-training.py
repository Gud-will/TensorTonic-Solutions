import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def binary_cross_entropy_loss(y: np.ndarray, p: np.ndarray, n: int):
    return -(1/n)*np.sum(y * np.log(p) + (1-y) * np.log(1-p))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # Write code here
    w = np.zeros(X.shape[1])
    b = 0
    while steps:
        z = X@w + b
        p = _sigmoid(z)
        loss = binary_cross_entropy_loss(y, p, X.shape[0])
        
        w_gradient = (1/X.shape[0])* X.T @(p-y)
        b_gradient = (1/X.shape[0])* np.sum(p-y)
        w = w - lr * w_gradient
        b = b- lr * b_gradient

        steps-=1
    return w,b

    
    