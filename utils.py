import numpy as np
def bit_entropy(codes):
    eps = 1e-40
    entropy = -(codes*codes.clamp(eps).log()+(1-codes)*(1-codes).clamp(eps).log())
    return entropy.mean()

def KNNdis(x,y):
    sum = 0
    for i in range(len(x)):
        sum = sum +(x[i]-y[i])**2
    return np.sqrt(sum)

def map(rec, pre):
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], pre, [0.]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]
    map = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return map

def hammingDistance(s1, s2):
    """Return the Hamming distance between equal-length sequences"""
    if len(s1) != len(s2):
        raise ValueError("Undefined for sequences of unequal length")
    return sum(el1 != el2 for el1, el2 in zip(s1, s2))