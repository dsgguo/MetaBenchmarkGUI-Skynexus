"""
使用猴子补丁修复CCA算法问题
"""

import numpy as np
from scipy.sparse import block_diag, identity, vstack
from scipy.linalg import qr


def _msetccar_kernel_fixed(X: np.ndarray, Yf: np.ndarray):
    """
    修复后的Multi-set CCA1 with reference signals (MsetCCA-R)内核函数
    
    修复内容：
    - 确保所有类别返回相同形状的T数组
    - 使用模板（均值）而不是所有试验来计算特征
    
    Parameters
    ----------
    X: ndarray
        EEG data, shape(n_trials, n_channels, n_samples).
    Yf: ndarray
        Reference signal, shape(n_classes, 2*n_harmonics, n_samples)
    """
    # 导入需要的函数
    from metabci.brainda.algorithms.decomposition.cca import _ged_wong
    
    X = np.reshape(X, (-1, *X.shape[-2:]))
    M, C, N = X.shape
    n_components = min(C, Yf.shape[0])
    P = vstack([identity(N) for _ in range(M)])
    Q, R = qr(Yf.T, mode="economic")
    P = P @ Q @ Q.T @ P.T
    Z = block_diag(X).T
    _, U = _ged_wong(Z, None, P, n_components=n_components)  # U for X
    
    # 修复：计算模板的特征而不是所有试验的特征，确保输出形状一致
    template = np.mean(X, axis=0)  # shape: (C, N)
    T = U.T @ template  # shape: (n_components, N)
    
    return U, T


def apply_cca_fix():
    """
    应用CCA修复补丁
    
    这个函数会替换原始的_msetccar_kernel函数
    """
    # 替换brainda包中的函数
    import metabci.brainda.algorithms.decomposition.cca as cca_module
    cca_module._msetccar_kernel = _msetccar_kernel_fixed
    
    # 替换benchmark包中的函数
    try:
        import metabci.benchmarkGUI.benchmark.algorithm.cca as benchmark_cca_module
        benchmark_cca_module._msetccar_kernel = _msetccar_kernel_fixed
    except ImportError:
        print("Warning: benchmark cca module not found, skipping patch")
    
    print("CCA修复补丁已应用")


if __name__ == "__main__":
    apply_cca_fix()
