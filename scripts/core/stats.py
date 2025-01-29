import numpy as np
try:
    import cupy as cp
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False
    cp = np
from scipy.stats import norm

def to_numpy(x):
    """Convert CuPy array to NumPy if needed."""
    if HAS_CUDA and isinstance(x, cp.ndarray):
        return x.get()
    return x

def significance_stars(p_value):
    """Convert p-value to significance stars."""
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    return ""

def test_accuracy(observed_acc, n_classes, n_samples):
    """One-Sample Proportion z-Test for accuracy."""
    observed_acc = to_numpy(observed_acc)
    p0 = 1.0 / n_classes
    z = (observed_acc - p0) / np.sqrt(p0 * (1 - p0) / n_samples)
    p_value = 1 - norm.cdf(z)
    return p_value

def test_gmcc(gmcc_value, n_samples):
    """One-Sample z-Test for GMCC (comparing to 0)."""
    if gmcc_value is None:
        return 1.0  # Return p=1 for None values
    gmcc_value = to_numpy(gmcc_value)
    z = gmcc_value * np.sqrt(n_samples)
    p_value = 1 - norm.cdf(z)
    return p_value

def test_frobenius(observed_fn, n_classes):
    """One-Sample z-Test for Frobenius norm."""
    observed_fn = to_numpy(observed_fn)
    expected_fn = n_classes**2 / 3
    var_fn = (4 * n_classes**2) / 45
    z = (observed_fn - expected_fn) / np.sqrt(var_fn)
    p_value = norm.cdf(z)  # Using cdf as we want FN < expected
    return p_value

def format_metric(value, p_value, format_str=".4f"):
    """Format metric with significance stars."""
    if value is None:
        return "N/A"
    value = to_numpy(value)
    stars = significance_stars(p_value)
    return f"{value:{format_str}}{stars}" 