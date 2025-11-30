
import numpy as np

DIM = 2**14  # 超向量维度

def generate_random_hypervector(dim=DIM):

    """Generate a random hypervector with elements 0 or 1."""
    return np.random.choice([0, 1], size = dim)

def bind(hv1, hv2):
    return np.logical_xor(hv1, hv2).astype(np.uint8)


def bundle(vectors, tie_break='one'):
    """
    Majority vote for binary hypervectors.
    
    Parameters:
        vectors: list of 1D numpy arrays or 2D numpy array (N, D)
        tie_break: 'one', 'zero', or 'random'
        
    Returns:
        1D numpy array of shape (D,), dtype=int
    """
    # If input is list, convert to matrix
    if isinstance(vectors, list):
        stacked = stack_vectors(vectors)
    else:
        stacked = np.array(vectors)  # assume already 2D array

    N, D = stacked.shape
    counts = np.sum(stacked, axis=0)  # number of 1s per column

    # majority vote
    out = (counts >= (N / 2)).astype(int)

    # handle ties if N is even
    if N % 2 == 0:
        ties = (counts == N / 2)
        if tie_break == 'random':
            out[ties] = np.random.randint(0, 2, size=np.sum(ties))
        elif tie_break == 'zero':
            out[ties] = 0
        # tie_break=='one' -> already 1, no action needed

    return out

def similarity(hv1, hv2):
    """Compute Hamming similarity (0~1) for 0/1 vectors."""
    return np.sum(hv1 == hv2) / len(hv1)


def stack_vectors(vectors):
    """
    Stack a list of 1D hypervectors into a 2D numpy array.
    
    Parameters:
        vectors: list or iterable of 1D numpy arrays of same length
        
    Returns:
        2D numpy array of shape (N, D)
    """
    if len(vectors) == 0:
        raise ValueError("vectors must be non-empty")
    
    return np.vstack(vectors)  # shape (N, D)