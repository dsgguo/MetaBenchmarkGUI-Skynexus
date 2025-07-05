# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2021/1/29
# License: MIT License

from typing import Optional, List, cast
from functools import partial

import numpy as np
from scipy.linalg import eigh, pinv, qr
from scipy.stats import pearsonr
from scipy.sparse import block_diag, identity, vstack, spmatrix
from scipy.sparse.linalg import eigsh

from numpy import ndarray
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.svm import SVC
from joblib import Parallel, delayed

from .base import FilterBankSSVEP


def _ged_wong(
    Z: ndarray,
    D: Optional[ndarray] = None,
    P: Optional[ndarray] = None,
    n_components=1,
    method="type1",
):
    if method != "type1" and method != "type2":
        raise ValueError("not supported method type")

    A = Z
    if D is not None:
        A = D.T @ A
    if P is not None:
        A = P.T @ A
    A = A.T @ A
    if method == "type1":
        B = Z
        if D is not None:
            B = D.T @ Z
        B = B.T @ B
        if isinstance(A, spmatrix) or isinstance(B, spmatrix):
            D, W = eigsh(A, k=n_components, M=B)
        else:
            D, W = eigh(A, B)
    elif method == "type2":
        if isinstance(A, spmatrix):
            D, W = eigsh(A, k=n_components)
        else:
            D, W = eigh(A)

    D_exist = cast(ndarray, D)
    ind = np.argsort(D_exist)[::-1]
    D_exist, W = D_exist[ind], W[:, ind]
    return D_exist[:n_components], W[:, :n_components]


def _scca_kernel(X: ndarray, Yf: ndarray):
    """Standard CCA (sCCA).

    This is an time-consuming implementation due to GED.

    X: (n_channels, n_samples)
    Yf: (n_harmonics, n_samples)
    """
    n_components = min(X.shape[0], Yf.shape[0])#根据`X`和`Yf`的行数（即特征数量）确定要提取的典型相关成分的最小值
    Q, R = qr(Yf.T, mode="economic")#QR分解是一种矩阵分解方法，其中`Q`是正交矩阵，`R`是上三角矩阵。这一步通常用于求解线性最小二乘问题或数据的正交化
    P = Q @ Q.T#1. 将`Q`与它的转置相乘得到一个投影矩阵`P`，这用于后续的数据转换。
    Z = X.T#将`X`矩阵转置，准备进行后续的计算。
    _, U = _ged_wong(Z, None, P, n_components=n_components)  # U for X,用于求解一般化的典型相关分析中的优化问题。返回的`U`是与`X`相关的典型载荷向量
    V = pinv(R) @ Q.T @ X.T @ U  # V for Yf
    return U, V


def _scca_feature(X: ndarray, Yf: ndarray, n_components: int = 1):
    rhos = []
    for Y in Yf:
        U, V = _scca_kernel(X, Y)
        a = U[:, :n_components].T @ X
        b = V[:, :n_components].T @ Y
        a = np.reshape(a, (-1))
        b = np.reshape(b, (-1))
        rhos.append(pearsonr(a, b)[0])#计算相关系数
    return np.array(rhos)


class SCCA(BaseEstimator, TransformerMixin, ClassifierMixin):
    """
    Standard CCA (sCCA).The Canonical Correlation Analysis (CCA) method finds the coefficients of the linear combination
    between the test signal and the Fourier series reference signal for a given frequency-periodic signal to find the
    maximum correlation between the two sets of signals. To identify the frequency of the SSVEP, CCA calculates the
    typical correlation between the multichannel SSVEP and the reference signal corresponding to each stimulus frequency,
    and the frequency of the reference signal with the largest correlation is regarded as the frequency of the
    SSVEP[1]_[2]_.SCCA is the standard CCA method.
    标准CCA（sCCA）。典型相关分析（CCA）方法找到给定频率周期信号的测试信号和傅里叶级数参考信号之间的线性组合系数，以找到两组信号之间的最大相关性。
    为了确定SSVEP的频率，CCA计算多通道SSVEP与每个刺激频率对应的参考信号之间的典型相关性，相关性最大的参考信号的频率被视为SSVEP[1]_[2]_的频率。
    SCCA是标准的CCA方法。

    Parameters
    ----------
    n_components : int
        The number of feature dimensions after dimensionality reduction,
        the dimension of the spatial filter, defaults to 1.
    n_jobs : int
        The number of CPU working cores, default is None.

    Attributes
    ----------
    Yf_ : ndarray
        The reference signal provided, defaults to None.

    Raises
    ----------
    ValueError
        None


    References
    ----------
    .. [1] Lin Z, Zhang C, Wu W, et al. Frequency recognition based on canonical correlation analysis for
        SSVEP-based BCIs[J].IEEE transactions on biomedical engineering, 2006, 53(12): 2610-2614.

    .. [2] Chen X, Wang Y, Nakanishi M, et al. High-speed spelling with a noninvasive brain–computer
        interface[J].Proceedings of the national academy of sciences, 2015, 112(44): E6058-E6067.

    Tip
    ----
    .. code-block:: python
       :caption: A example using SCCA

        from metabci.brainda.algorithms.decomposition.cca import SCCA
        estimator = SCCA()
        p_labels = estimator.fit(X=X[train_ind],y=y[train_ind], Yf=Yf).predict(X[test_ind])

    """
    def __init__(self, n_components: int = 1, n_jobs: Optional[int] = None):
        self.n_components = n_components
        self.n_jobs = n_jobs

    def fit(
        self,
        X: Optional[ndarray] = None,
        y: Optional[ndarray] = None,
        Yf: Optional[ndarray] = None,
    ):
        """ model training

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).
        y: ndarray
            Label, shape(n_trials,)
        Yf: ndarray
            Sine and cosine reference signal, shape(n_classes, 2*n_harmonics, n_samples).
        """
        if Yf is None:
            raise ValueError("The reference signals Yf should be provided.")
        Yf = np.reshape(Yf, (-1, *Yf.shape[-2:]))
        Yf = Yf - np.mean(Yf, axis=-1, keepdims=True)
        self.Yf_ = Yf
        return self

    def transform(self, X: ndarray):
        """The correlation coefficients of the signals from different trials were obtained by converting X
        into features.

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).

        Returns
        ----------
        rhos: ndarray
            he correlation coefficients, shape(n_trials, n_fre)

        """
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        Yf = self.Yf_
        # print(Yf.shape, X.shape)
        n_components = self.n_components
        rhos = Parallel(n_jobs=self.n_jobs)(
            delayed(partial(_scca_feature, n_components=n_components))(a, Yf) for a in X
        )
        rhos = np.stack(rhos)
        return rhos

    def predict(self, X: ndarray):
        """Predict the labels

        Parameters
        ----------
        X : ndarray
            EEG data, shape(n_trials, n_channels, n_samples).

        Returns
        ----------
        labels : ndarray
            Predicting labels, shape(n_trials,).
        """
        rhos = self.transform(X)
        labels = np.argmax(rhos, axis=-1)#沿着最后一个轴找最大索引
        return labels


class FBSCCA(FilterBankSSVEP, ClassifierMixin):
    """
    Filter bank SCCA methods, i.e., SCCA methods that combine the application of multiple filters in order to decompose
    the SSVEP signal into specific subbands[1]_ .This class is a FBSCCA classifier.

    Parameters
    ----------
    filterbank: list[ndarray]
        Filter bank list
    n_components: int
        The number of feature dimensions after dimensionality reduction, the dimension of the spatial filter,
        defaults to 1.
    filterweights: ndarray
        Filter weights, defaults to None.
    n_jobs: int
        The number of CPU working cores, default is None.

    References
    ----------
    .. [1] Chen X, Wang Y, Gao S, et al. Filter bank canonical correlation analysis for implementing a high-speed
        SSVEP-based brain–computer interface[J]. Journal of neural engineering, 2015, 12(4): 046008.

    Tip
    ----
    .. code-block:: python
       :linenos:
       :caption: A example using FBSCCA

       import sys
       import numpy as np
       from brainda.algorithms.decomposition import FBSCCA
       from brainda.algorithms.decomposition.base import generate_filterbank, generate_cca_references
       wp=[(5,90),(14,90),(22,90),(30,90),(38,90)]
       ws=[(3,92),(12,92),(20,92),(28,92),(36,92)]
       filterbank = generate_filterbank(wp,ws,srate=250,order=15,rp=0.5)
       filterweights = [(idx_filter+1) ** (-1.25) + 0.25 for idx_filter in range(5)]
       estimator = FBSCCA(filterbank=filterbank,n_components=1,filterweights=np.array(filterweights),n_jobs=-1)
       accs = []
       for k in range(kfold):
           train_ind, validate_ind, test_ind = match_kfold_indices(k, meta, indices)
           # merge train and validate set
           train_ind = np.concatenate((train_ind, validate_ind))
           p_labels = estimator.fit(X=X[train_ind],y=y[train_ind], Yf=Yf).predict(X[test_ind])
           accs.append(np.mean(p_labels==y[test_ind]))
           print(np.mean(accs))
    """
    def __init__(
        self,
        filterbank: List[ndarray],
        n_components: int = 1,
        filterweights: Optional[ndarray] = None,
        n_jobs: Optional[int] = None,
    ):
        self.n_components = n_components
        self.filterweights = filterweights
        self.n_jobs = n_jobs
        super().__init__(
            filterbank,
            SCCA(n_components=n_components, n_jobs=1),
            filterweights=filterweights,
            n_jobs=n_jobs,
        )

    def predict(self, X: ndarray):
        """Predict the labels

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).

        Returns
        ----------
        labels: ndarray
            Predicting labels, shape(n_trials,).
        """
        features = self.transform(X)
        if self.filterweights is None:
            features = np.reshape(
                features, (features.shape[0], len(self.filterbank), -1)
            )
            features = np.mean(features, axis=1)
        labels = np.argmax(features, axis=-1)
        return labels


def _itcca_feature(
    X: ndarray,
    templates: ndarray,
    Us: Optional[ndarray] = None,
    n_components: int = 1,
    method: str = "itcca1",
):
    """
    ItCCA feature extraction
    """
    rhos = []
    if method == "itcca1":
        for Xk in templates:
            U, V = _scca_kernel(X, Xk)
            a = U[:, :n_components].T @ X
            b = V[:, :n_components].T @ Xk
            a, b = np.reshape(a, (-1)), np.reshape(b, (-1))
            rhos.append(pearsonr(a, b)[0])
    elif method == "itcca2":
        Us = cast(ndarray, Us)
        for Xk, U in zip(templates, Us):
            a = U[:, :n_components].T @ X
            b = U[:, :n_components].T @ Xk
            a, b = np.reshape(a, (-1)), np.reshape(b, (-1))
            rhos.append(pearsonr(a, b)[0])
    return np.array(rhos)


class ItCCA(BaseEstimator, TransformerMixin, ClassifierMixin):
    """
    The Individual Template-based Canonical Correlation Analysis (It-CCA) method is an extension of the CCA method in
    which the reference signal is a VEP template obtained by averaging multiple EEG trials from each individual's
    calibration data, and the individual SSVEP training data is used in the CCA method to improve the frequency detection
    of SSVEP [1]_.This class is a itCCA classifier

    Parameters
    ----------
    n_components: int
        The number of feature dimensions after dimensionality reduction, the dimension of the spatial filter,
        defaults to 1.
    method: str
        Two pattern feature extraction and fitting classifier model methods judgment, defaulting to 'itcca2'.
    n_jobs: int
        The number of CPU working cores, default is None.

    Attributes
    ----------
    Yf_: ndarray
        Reference signal.
    classes_ : ndarray
        Predictive labels, obtained from labeled data by removing duplicate elements from it.
    templates_: ndarray
        Test data after spatial filtering
    Us_: ndarray
        Spatial filter
    Vs_: ndarray
        Spatial filter

    References
    ----------
    .. [1] Brogin J A F, Faber J, Bueno D D. Enhanced use practices in SSVEP-based BCIs using an analytical approach of
        canonical correlation analysis[J]. Biomedical Signal Processing and Control, 2020, 55: 101644.
    """
    def __init__(
        self,
        n_components: int = 1,
        method: str = "itcca2",
        n_jobs: Optional[int] = None,
    ):
        self.n_components = n_components
        self.method = method
        self.n_jobs = n_jobs

    def fit(self, X: ndarray, y: ndarray, Yf: Optional[ndarray] = None):
        """model train

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).
        y: ndarray
            Labels, shape(n_trials,)
        Yf: ndarray
            Reference signal(n_classes, 2*n_harmonics, n_samples)
        """
        if self.method == "itcca2" and Yf is None:
            raise ValueError("The reference signals Yf should be provided.")
        self.classes_ = np.unique(y)
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        self.templates_ = np.stack(
            [np.mean(X[y == label], axis=0) for label in self.classes_]
        )
        if self.method == "itcca2":
            Yf = cast(ndarray, Yf)
            Yf = np.reshape(Yf, (-1, *Yf.shape[-2:]))
            Yf = Yf - np.mean(Yf, axis=-1, keepdims=True)
            self.Yf_ = cast(ndarray, Yf)
            self.Us_, self.Vs_ = zip(
                *[
                    _scca_kernel(self.templates_[i], self.Yf_[i])
                    for i in range(len(self.classes_))
                ]
            )
            self.Us_, self.Vs_ = np.stack(self.Us_), np.stack(self.Vs_)
        return self

    def transform(self, X: ndarray):
        """Transform the X into features and calculate the correlation coefficients of different trials

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).

        Returns
        ----------
        rhos: ndarray
            Correlation coefficients, shape(n_trials, n_fre).

        """
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        templates = self.templates_
        n_components = self.n_components
        method = self.method
        Us = None
        if method == "itcca2":
            Us = self.Us_
        rhos = Parallel(n_jobs=self.n_jobs)(
            delayed(
                partial(_itcca_feature, Us=Us, n_components=n_components, method=method)
            )(a, templates)
            for a in X
        )
        rhos = np.stack(rhos)
        return rhos

    def predict(self, X: ndarray):
        """Predict the labels

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).

        Returns
        ----------
        labels: ndarray
            Predicting labels, shape(n_trials,).
        """
        rhos = self.transform(X)
        labels = self.classes_[np.argmax(rhos, axis=-1)]
        return labels


class FBItCCA(FilterBankSSVEP, ClassifierMixin):
    """
    The filter bank ItCCA method, i.e., the ItCCA method that combines the application of multiple filters in order
    to decompose the SSVEP signal into specific subbands[1]_.

    Parameters
    ----------
    filterbank: list[ndarray]
        Filter bank list
    n_components: int
        The number of feature dimensions after dimensionality reduction, the dimension of the spatial filter,
        defaults to 1.
    filterweights: ndarray
        Filter weights, defaults to None.
    n_jobs: int
        The number of CPU working cores, default is None.
    method: str
        Two pattern feature extraction and fitting classifier model methods judgment, defaulting to 'itcca2'.

    Attributes
    ----------
    classes_ : ndarray
        Predictive labels, obtained from labeled data by removing duplicate elements from it.

    References
    ----------
    .. [1] Bolaños M C, Ballestero S B, Puthusserypady S. Filter bank approach for enhancement of supervised Canonical
        Correlation Analysis methods for SSVEP-based BCI spellers[C]//2021 43rd Annual International Conference of
        the IEEE Engineering in Medicine & Biology Society (EMBC). IEEE, 2021: 337-340.

    """
    def __init__(
        self,
        filterbank: List[ndarray],
        n_components: int = 1,
        method: str = "itcca2",
        filterweights: Optional[ndarray] = None,
        n_jobs: Optional[int] = None,
    ):
        self.n_components = n_components
        self.method = method
        self.filterweights = filterweights
        self.n_jobs = n_jobs
        super().__init__(
            filterbank,
            ItCCA(n_components=n_components, method=method, n_jobs=1),
            filterweights=filterweights,
            n_jobs=n_jobs,
        )

    def fit(self, X: ndarray, y: ndarray, Yf: Optional[ndarray] = None):  # type: ignore[override]
        """model train

       Parameters
       ----------
       X: ndarray
           EEG data, shape(n_trials, n_channels, n_samples).
       y: ndarray
           Labels, shape(n_trials,)
       Yf: ndarray
           Reference signal(n_classes, 2*n_harmonics, n_samples)
        """
        self.classes_ = np.unique(y)
        super().fit(X, y, Yf=Yf)
        return self

    def predict(self, X: ndarray):
        """Predict the labels

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).

        Returns
        ----------
        labels: ndarray
            Predicting labels, shape(n_trials,).
        """
        features = self.transform(X)
        if self.filterweights is None:
            features = np.reshape(
                features, (features.shape[0], len(self.filterbank), -1)
            )
            features = np.mean(features, axis=1)
        labels = self.classes_[np.argmax(features, axis=-1)]
        return labels


def _mscca_feature(X: ndarray, templates: ndarray, U: ndarray, n_components: int = 1):
    rhos = []
    for Xk in zip(templates):
        a = U[:, :n_components].T @ X
        b = U[:, :n_components].T @ Xk
        a, b = np.reshape(a, (-1)), np.reshape(b, (-1))
        rhos.append(pearsonr(a, b)[0])
    return np.array(rhos)


class MsCCA(BaseEstimator, TransformerMixin, ClassifierMixin):
    """
    Since the sine-cosine signal may not be the ideal reference signal, the Multiset Canonical Correlation Analysis
    (MsetCCA) method uses joint spatial filtering of multiple sets of data to create an optimized reference signal that
    extracts common SSVEP features from multiple sets of EEG data recorded at the same stimulus frequency[1]_.
    Note: MsCCA heavily depends on Yf, thus the phase information should be included when designs Yf.

    Parameters
    ----------
    n_components: int
        The number of feature dimensions after dimensionality reduction, the dimension of the spatial filter,
        defaults to 1.
    method: str
        Two pattern feature extraction and fitting classifier model methods judgment, defaulting to 'itcca2'.
    n_jobs: int
        The number of CPU working cores, default is None.

    Attributes
    ----------
    classes_ : ndarray
        Predictive labels, obtained from labeled data by removing duplicate elements from it.
    Yf_: ndarray
        Reference signals
    Us_: ndarray
        Spatial filter
    Ts_: ndarray
        Spatial filter

    References
    ----------
    .. [1] Zhang YU, Zhou G, Jin J, et al. Frequency recognition in SSVEP-based BCI using multiset canonical correlation
        analysis[J]. International journal of neural systems, 2014, 24(04): 1450013.
    """

    def __init__(self, n_components: int = 1, n_jobs: Optional[int] = None):
        self.n_components = n_components
        self.n_jobs = n_jobs

    def fit(self, X: ndarray, y: ndarray, Yf: ndarray):
        """model train

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).
        y: ndarray
            Labels, shape(n_trials,)
        Yf: ndarray
            Reference signal(n_classes, 2*n_harmonics, n_samples)
        """

        self.classes_ = np.unique(y)
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        self.templates_ = np.stack(
            [np.mean(X[y == label], axis=0) for label in self.classes_]
        )

        Yf = np.reshape(Yf, (-1, *Yf.shape[-2:]))
        Yf = Yf - np.mean(Yf, axis=-1, keepdims=True)
        self.Yf_ = Yf

        self.U_, self.V_ = _scca_kernel(
            np.concatenate(self.templates_, axis=-1), np.concatenate(self.Yf_, axis=-1)
        )
        return self

    def transform(self, X: ndarray):
        """Transform X into features and calculate the correlation coefficients of
        the signals from different trials.

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).

        Returns
        ----------
        rhos: ndarray
            The correlation coefficients, shape(n_trials, n_fre).
        """
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        templates = self.templates_
        n_components = self.n_components
        U = self.U_
        rhos = Parallel(n_jobs=self.n_jobs)(
            delayed(partial(_mscca_feature, U=U, n_components=n_components))(
                a, templates
            )
            for a in X
        )
        rhos = np.stack(rhos)
        return rhos

    def predict(self, X: ndarray):
        """Predict the labels

            Parameters
            ----------
            X: ndarray
                EEG data, shape(n_trials, n_channels, n_samples).

            Returns
            ----------
            labels: ndarray
                Predicting labels, shape(n_trials,).
        """
        rhos = self.transform(X)
        labels = self.classes_[np.argmax(rhos, axis=-1)]
        return labels


class FBMsCCA(FilterBankSSVEP, ClassifierMixin):
    """
    The filter bank MsetCCA method, i.e., the MsetCCA method that combines the application of multiple filters
    in order to decompose the SSVEP signal into specific subbands[1]_.

    Parameters
    ----------
    filterbank: list[ndarray]
        Filter bank list.
    filterweights: ndarray
        Weights of filter banks
    n_components: int
        The number of feature dimensions after dimensionality reduction, the dimension of the spatial filter,
        defaults to 1.
    method: str
        Two pattern feature extraction and fitting classifier model methods judgment, defaulting to 'itcca2'.
    n_jobs: int
        The number of CPU working cores, default is None.

    Attributes
    ----------
    classes_ : ndarray
        Predictive labels, obtained from labeled data by removing duplicate elements from it.

    References
    ----------
    .. [1] Zhang Y U, Zhou G, Jin J, et al. Frequency recognition in SSVEP-based BCI using multiset canonical correlation
        analysis[J]. International journal of neural systems, 2014, 24(04): 1450013.
     """
    def __init__(
        self,
        filterbank: List[ndarray],
        n_components: int = 1,
        filterweights: Optional[ndarray] = None,
        n_jobs: Optional[int] = None,
    ):
        self.n_components = n_components
        self.filterweights = filterweights
        self.n_jobs = n_jobs
        super().__init__(
            filterbank,
            MsCCA(n_components=n_components, n_jobs=1),
            filterweights=filterweights,
            n_jobs=n_jobs,
        )

    def fit(self, X: ndarray, y: ndarray, Yf: Optional[ndarray] = None):  # type: ignore[override]
        """model train

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).
        y: ndarray
            Labels, shape(n_trials,)
        Yf: ndarray
            Reference signal(n_classes, 2*n_harmonics, n_samples)
        """

        self.classes_ = np.unique(y)
        super().fit(X, y, Yf=Yf)
        return self

    def predict(self, X: ndarray):
        """Predict the labels

            Parameters
            ----------
            X: ndarray
                EEG data, shape(n_trials, n_channels, n_samples).

            Returns
            ----------
            labels: ndarray
                Predicting labels, shape(n_trials,).
        """
        features = self.transform(X)
        if self.filterweights is None:
            features = np.reshape(
                features, (features.shape[0], len(self.filterbank), -1)
            )
            features = np.mean(features, axis=1)
        labels = self.classes_[np.argmax(features, axis=-1)]
        return labels


def _ecca_feature(
    X: ndarray,
    templates: ndarray,
    Yf: ndarray,
    Us: Optional[ndarray] = None,
    n_components: int = 1,
):
    if Us is None:
        Us_array, _ = zip(
            *[_scca_kernel(templates[i], Yf[i]) for i in range(len(templates))]
        )
        Us = np.stack(Us_array)
    rhos = []
    for Xk, Y, U3 in zip(templates, Yf, Us):
        rho_list = []
        # 14a, 14d
        U1, V1 = _scca_kernel(X, Y)
        a = U1[:, :n_components].T @ X
        b = V1[:, :n_components].T @ Y
        a, b = np.reshape(a, (-1)), np.reshape(b, (-1))
        rho_list.append(pearsonr(a, b)[0])
        a = U1[:, :n_components].T @ X
        b = U1[:, :n_components].T @ Xk
        a, b = np.reshape(a, (-1)), np.reshape(b, (-1))
        rho_list.append(pearsonr(a, b)[0])
        # 14b
        U2, _ = _scca_kernel(X, Xk)
        a = U2[:, :n_components].T @ X
        b = U2[:, :n_components].T @ Xk
        a, b = np.reshape(a, (-1)), np.reshape(b, (-1))
        rho_list.append(pearsonr(a, b)[0])
        # 14c
        a = U3[:, :n_components].T @ X
        b = U3[:, :n_components].T @ Xk
        a, b = np.reshape(a, (-1)), np.reshape(b, (-1))
        rho_list.append(pearsonr(a, b)[0])
        rho = np.array(rho_list)
        rho = np.sum(np.sign(rho) * (rho**2))
        rhos.append(rho)
    return rhos


class ECCA(BaseEstimator, TransformerMixin, ClassifierMixin):
    """The Extended Canonical Correlation Analysis (eCCA) method combines the advantages of sCCA and itCCA while
    applying the individual averaging templates and the positive cosine reference signal correlation information,
    thus obtaining better recognition performance[1]_.

    Parameters
    ----------
    n_components: int
        The number of feature dimensions after dimensionality reduction, the dimension of the spatial filter,
        defaults to 1.
    n_jobs: int
        The number of CPU working cores, default is None.

    Attributes
    ----------
    classes_ : ndarray
        Predictive labels, obtained from labeled data by removing duplicate elements from it.
        预测标签，通过从标记数据中删除重复元素而获得。
    Yf_: ndarray
        Reference signals.
    Us_: ndarray
        Spatial filter.
    Vs_: ndarray
        Spatial filter.

    References
    ----------
    .. [1] Chen X, Wang Y, Nakanishi M, et al. High-speed spelling with a noninvasive brain–computer interface[J].
        Proceedings of the national academy of sciences. 2015. 112(44): E6058-E6067.
    """
    def  __init__(self, n_components: int = 1, n_jobs: Optional[int] = None):
        self.n_components = n_components
        self.n_jobs = n_jobs

    def fit(self, X: ndarray, y: ndarray, Yf: ndarray):
        """model train

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).
        y: ndarray
            Labels, shape(n_trials,).
        Yf: ndarray
            Reference signal(n_classes, 2*n_harmonics, n_samples).
        """

        self.classes_ = np.unique(y)
        X = np.reshape(X, (-1, *X.shape[-2:]))#如果X原本的形状是 (n, m, h, w)，重塑后可能变为 (n*m, h, w)
        X = X - np.mean(X, axis=-1, keepdims=True)
        self.templates_ = np.stack(
            [np.mean(X[y == label], axis=0) for label in self.classes_]#这样可以保留特征空间的维度，而将不同样本的同一特征平均化
        )

        Yf = np.reshape(Yf, (-1, *Yf.shape[-2:]))
        Yf = Yf - np.mean(Yf, axis=-1, keepdims=True)
        self.Yf_ = Yf
        self.Us_, self.Vs_ = zip(#输出是对应于每个类别的U和V矩阵（或者向量），这些是CCA中的关键成分，代表了输入数据在降维空间中的表示。
            *[
                _scca_kernel(self.templates_[i], self.Yf_[i])
                for i in range(len(self.classes_))
            ]
        )
        self.Us_, self.Vs_ = np.stack(self.Us_), np.stack(self.Vs_)
        return self#这通常意味着这个方法是类的一个构造器或者是一个对实例状态有修改的处理方法，并且支持链式调用。

    def transform(self, X: ndarray):
        """Transform X into features and calculate the correlation coefficients of
        the signals from different trials
        将X转换为特征，并计算不同试验信号的相关系数
        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).

        Returns
        ----------
        rhos: ndarray
            The correlation coefficients, shape(n_trials, n_fre)
        """
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        templates = self.templates_
        Yf = self.Yf_
        Us = self.Us_
        n_components = self.n_components
        # `partial` 函数来自 `functools` 库，用于创建一个新的函数，这个新函数的部分参数已经被固定（在这里是 `Us` 和 `n_components`）。
        # 这意味着 `_ecca_feature` 函数被部分参数预先绑定，只等待剩下的参数在并行执行时传入。
        #`delayed` 则是让 `_ecca_feature` 函数及其预设参数适合于在 `Parallel` 环境下延迟执行，即任务调度时不会立即计算，而是在并行工作时按需计算。
        rhos = Parallel(n_jobs=self.n_jobs)(
            delayed(partial(_ecca_feature, Us=Us, n_components=n_components))(
                a, templates, Yf#对于 `X` 中的每一个元素 `a`，这个循环构造了一个元组 `(a, templates, Yf)`，作为 `_ecca_feature` 函数的剩余参数。
            )
            for a in X
        )
        rhos = np.stack(rhos)
        return rhos

    def predict(self, X: ndarray):
        """Predict the labels

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).

        Returns
        ----------
        labels: ndarray
            Predicting labels, shape(n_trials,).
        """
        rhos = self.transform(X)
        #`沿着 `rhos` 的最后一个轴（axis=-1）找到最大值的索引，即对于每个样本，找出哪个类别具有最高相关性或匹配度。`
        labels = self.classes_[np.argmax(rhos, axis=-1)]
        return labels


class FBECCA(FilterBankSSVEP, ClassifierMixin):
    """Filter bank eCCA method, i.e., an eCCA method that combines the application of multiple filters in order
    to decompose the SSVEP signal into specific subbands [1]_.

    Parameters
    ----------
    filterbank: list[ndarray]
        Filter bank list
    filterweights: ndarray
        Weights of filter bank
    n_components: int
        降维后的特征维数、空间滤波器的维数，
        The number of feature dimensions after dimensionality reduction, the dimension of the spatial filter,
        defaults to 1.
    n_jobs: int
        The number of CPU working cores, default is None.

    Attributes
    ----------
    classes_ : ndarray
        Predictive labels, obtained from labeled data by removing duplicate elements from it.
        预测标签，通过从标记数据中删除重复元素而获得。
    References
    ----------
    .. [1] Tong C, Wang H. A Novel Low Training Cost SSVEP Detector Design[C]//2021 14th International Symposium on
        Computational Intelligence and Design (ISCID). IEEE, 2021: 130-133.

    Tip
    ----
    .. code-block:: python
       :linenos:
       :caption: A example using FBECCA

       import sys
       import numpy as np
       from brainda.algorithms.decomposition import FBECCA
       from brainda.algorithms.decomposition.base import generate_filterbank, generate_cca_references
       wp=[(5,90),(14,90),(22,90),(30,90),(38,90)]
       ws=[(3,92),(12,92),(20,92),(28,92),(36,92)]
       filterbank = generate_filterbank(wp,ws,srate=250,order=15,rp=0.5)
       filterweights = [(idx_filter+1) ** (-1.25) + 0.25 for idx_filter in range(5)]
       estimator = FBECCA(filterbank=filterbank,n_components=1,filterweights=np.array(filterweights),n_jobs=-1)
       accs = []
       for k in range(kfold):
            train_ind, validate_ind, test_ind = match_kfold_indices(k, meta, indices)
            train_ind = np.concatenate((train_ind, validate_ind))
            p_labels = estimator.fit(X=X[train_ind],y=y[train_ind], Yf=Yf).predict(X[test_ind])
            accs.append(np.mean(p_labels==y[test_ind]))
       print(np.mean(accs))
    """
    def __init__(
        self,
        filterbank: List[ndarray],
        n_components: int = 1,
        filterweights: Optional[ndarray] = None,
        n_jobs: Optional[int] = None,
    ):
        self.n_components = n_components
        self.filterweights = filterweights
        self.n_jobs = n_jobs
        super().__init__(
            filterbank,
            ECCA(n_components=n_components, n_jobs=1),
            filterweights=filterweights,
            n_jobs=n_jobs,
        )

    def fit(self, X: ndarray, y: ndarray, Yf: Optional[ndarray] = None):  # type: ignore[override]
        """model train

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).
        y: ndarray
            Labels, shape(n_trials,)
        Yf: ndarray
            Reference signal(n_classes, 2*n_harmonics, n_samples)
        """

        self.classes_ = np.unique(y)
        super().fit(X, y, Yf=Yf)
        return self

    def predict(self, X: ndarray):
        """Predict the labels

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).

        Returns
        ----------
        labels: ndarray
            Predicting labels, shape(n_trials,).
        """
        features = self.transform(X)
        if self.filterweights is None:
            features = np.reshape(
                features, (features.shape[0], len(self.filterbank), -1)
            )
            features = np.mean(features, axis=1)
        labels = self.classes_[np.argmax(features, axis=-1)]
        return labels


def _ttcca_template(X: ndarray, y: ndarray, y_sub: Optional[ndarray] = None):
    X = np.reshape(X, (-1, *X.shape[-2:]))
    X = X - np.mean(X, axis=-1, keepdims=True)
    labels = np.unique(y)
    if y_sub is None:
        templates = np.stack([np.mean(X[y == label], axis=0) for label in labels])
    else:
        subjects = np.unique(y_sub)
        templates = 0
        for sub_id in subjects:
            templates += np.stack(
                [
                    np.mean(X[(y == label) & (y_sub == sub_id)], axis=0)
                    for label in labels
                ]
            )
        templates /= len(subjects)
    return templates


def _ttcca_feature(
    X: ndarray,
    templates: ndarray,
    Yf: ndarray,
    Us: Optional[ndarray] = None,
    n_components: int = 1,
):
    if Us is None:
        Us_array, _ = zip(
            *[_scca_kernel(templates[i], Yf[i]) for i in range(len(templates))]
        )
        Us = np.stack(Us_array)
    rhos = []
    for Xk, Y, U2 in zip(templates, Yf, Us):
        rho_list = []
        # rho1
        U1, V1 = _scca_kernel(X, Y)
        a = U1[:, :n_components].T @ X
        b = V1[:, :n_components].T @ Y
        a, b = np.reshape(a, (-1)), np.reshape(b, (-1))
        rho_list.append(pearsonr(a, b)[0])
        # rho3
        a = U1[:, :n_components].T @ X
        b = U1[:, :n_components].T @ Xk
        a, b = np.reshape(a, (-1)), np.reshape(b, (-1))
        rho_list.append(pearsonr(a, b)[0])
        # rho2
        a = U2[:, :n_components].T @ X
        b = U2[:, :n_components].T @ Xk
        a, b = np.reshape(a, (-1)), np.reshape(b, (-1))
        rho_list.append(pearsonr(a, b)[0])
        rho = np.array(rho_list)
        rho = np.sum(np.sign(rho) * (rho**2))
        rhos.append(rho)
    return rhos


class TtCCA(BaseEstimator, TransformerMixin, ClassifierMixin):
    """
    The Transfer Template-based Canonical Correlation Analysis (tt-CCA) method migrates SSVEP templates from existing
    subjects to new subjects to enhance SSVEP detection. EEG templates were generated for the new subjects using the
    existing source subject dataset, i.e., migrating EEG templates to capture the frequency and phase information of
    SSVEP[1]_.

    Parameters
    ----------
    n_components: int
        The number of feature dimensions after dimensionality reduction, the dimension of the spatial filter,
        defaults to 1.
    n_jobs: int
        The number of CPU working cores, default is None.

    Attributes
    ----------
    classes_ : ndarray
        Predictive labels, obtained from labeled data by removing duplicate elements from it.
    templates_: ndarray
        Individual average template signals.
    Yf_: ndarray
        Reference signals.
    Us_: ndarray
        Spatial filter.
    Vs_: ndarray
        Spatial filter.

    References
    ----------
    .. [1] Yuan P, Chen X, Wang Y, et al. Enhancing performances of SSVEP-based brain–computer interfaces via exploiting
        inter-subject information[J]. Journal of neural engineering, 2015, 12(4): 046006.

    """
    def __init__(self, n_components: int = 1, n_jobs: Optional[int] = None):
        self.n_components = n_components
        self.n_jobs = n_jobs

    def fit(self, X: ndarray, y: ndarray, Yf: ndarray, y_sub: Optional[ndarray] = None):
        """model train

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).
        y: ndarray
            Labels, shape(n_trials,)
        Yf: ndarray
            Reference signal(n_classes, 2*n_harmonics, n_samples)
        y_sub: ndarray
            Existing source subject data(X_sub.shape == (n_source_trials, n_channels, n_samples) or y_sub.shape == (n_source_trials,))
        """

        self.classes_ = np.unique(y)
        self.templates_ = _ttcca_template(X, y, y_sub=y_sub)

        Yf = np.reshape(Yf, (-1, *Yf.shape[-2:]))
        Yf = Yf - np.mean(Yf, axis=-1, keepdims=True)
        self.Yf_ = Yf
        self.Us_, self.Vs_ = zip(
            *[
                _scca_kernel(self.templates_[i], self.Yf_[i])
                for i in range(len(self.classes_))
            ]
        )
        return self

    def transform(self, X: ndarray):
        """Transform X into features and calculate the correlation coefficients of the signals from different trials

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).

        Returns
        ----------
        rhos: ndarray
            The correlation coefficients, shape(n_trials, n_fre)
        """
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        templates = self.templates_
        Yf = self.Yf_
        Us = self.Us_
        n_components = self.n_components
        rhos = Parallel(n_jobs=self.n_jobs)(
            delayed(partial(_ttcca_feature, Us=Us, n_components=n_components))(
                a, templates, Yf
            )
            for a in X
        )
        rhos = np.stack(rhos)
        return rhos

    def predict(self, X: ndarray):
        """Predict the labels

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).

        Returns
        ----------
        labels: ndarray
            Predicting labels, shape(n_trials,).
        """
        rhos = self.transform(X)
        labels = self.classes_[np.argmax(rhos, axis=-1)]
        return labels


class FBTtCCA(FilterBankSSVEP, ClassifierMixin):
    """Filter bank TtCCA method, i.e., a TtCCA method that combines the application of multiple filters in order to
    decompose the SSVEP signal into specific subbands[1]_.

    Parameters
    ----------
    filterbank: list[ndarray]
        Filter bank list
    filterweights: ndarray
        Weights of filter banks
    n_components: int
        The number of feature dimensions after dimensionality reduction, the dimension of the spatial filter,
        defaults to 1.
    n_jobs: int
        The number of CPU working cores, default is None.

    Attributes
    ----------
    classes_ : ndarray
        Predictive labels, obtained from labeled data by removing duplicate elements from it.


    References
    ----------
    .. [1] Yuan P, Chen X, Wang Y, et al. Enhancing performances of SSVEP-based brain–computer interfaces via
        exploiting inter-subject information[J]. Journal of neural engineering, 2015, 12(4): 046006.


    """
    def __init__(
        self,
        filterbank: List[ndarray],
        n_components: int = 1,
        filterweights: Optional[ndarray] = None,
        n_jobs: Optional[int] = None,
    ):
        self.n_components = n_components
        self.filterweights = filterweights
        self.n_jobs = n_jobs
        super().__init__(
            filterbank,
            TtCCA(n_components=n_components, n_jobs=1),
            filterweights=filterweights,
            n_jobs=n_jobs,
        )

    def fit(self, X: ndarray,  # type: ignore[override]
            y: ndarray,
            Yf: Optional[ndarray] = None,
            y_sub: Optional[ndarray] = None):
        """model train

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).
        y: ndarray
            Labels, shape(n_trials,).
        Yf: ndarray
            Reference signal(n_classes, 2*n_harmonics, n_samples).
        y_sub: ndarray
            Existing source subject data.
        """

        self.classes_ = np.unique(y)
        super().fit(X, y, Yf=Yf, y_sub=y_sub)
        return self

    def predict(self, X: ndarray):
        """Predict the labels

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).

        Returns
        ----------
        labels: ndarray
            Predicting labels, shape(n_trials,).
        """
        features = self.transform(X)
        if self.filterweights is None:
            features = np.reshape(
                features, (features.shape[0], len(self.filterbank), -1)
            )
            features = np.mean(features, axis=1)
        labels = self.classes_[np.argmax(features, axis=-1)]
        return labels


def _msetcca_kernel1(X: ndarray):
    """Multi-set CCA1 (MsetCCA1).

    X: (n_trials, n_channels, n_samples)
    """
    X = np.reshape(X, (-1, *X.shape[-2:]))
    M, C, N = X.shape
    n_components = C
    P = vstack([identity(N) for _ in range(M)])
    P = P @ P.T
    Z = block_diag(X).T
    _, U = _ged_wong(Z, None, P, n_components=n_components)  # U for X
    X = np.reshape(X, (-1, N))
    T = U.T @ X
    return U, T


def _msetcca_kernel2(X: ndarray, Xk: ndarray, Yf: ndarray):
    C, N = X.shape
    M = 3
    n_components = C
    P = vstack([identity(N) for _ in range(M)])
    P = P @ P.T
    Z = block_diag([X, Yf, Xk]).T
    D, U = _ged_wong(Z, None, P, n_components=n_components)  # U for X
    return U[:C, :], D


def _msetcca_feature2(
    X: ndarray, templates: ndarray, Yf: ndarray, n_components: int = 1
):
    feat = []
    for Xk, Y in zip(templates, Yf):
        U, D = _msetcca_kernel2(X, Xk, Y)
        A = U[:, :n_components].T @ X
        B = U[:, :n_components].T @ Xk
        rho = np.array([pearsonr(a, b)[0] for a, b in zip(A, B)])
        rho = D[0] * np.sign(rho) * (rho**2)
        feat.append(rho)
    feat = np.concatenate(feat, axis=-1)
    return feat


class MsetCCA(BaseEstimator, TransformerMixin, ClassifierMixin):
    """Since the sine-cosine signal may not be the ideal reference signal, the Multiset Canonical Correlation Analysis
    (MsetCCA) method uses joint spatial filtering of multiple sets of data to create an optimized reference signal that
    extracts common SSVEP features from multiple sets of EEG data recorded at the same stimulus frequency[1]_.

    Parameters
    ----------
    n_components: int
        The number of feature dimensions after dimensionality reduction, the dimension of the spatial filter,
        defaults to 1.
    n_jobs: int
        The number of CPU working cores, default is None.
    methods: str
        Two Pattern Feature Extraction and Fitting Classifier Model Methods Judgment, defaulting to 'msetcca2'.

    Attributes
    ----------
    classes_ : ndarray
        Predictive labels, obtained from labeled data by removing duplicate elements from it.
    templates_: ndarray
        Template signals
    Yf_: ndarray
        Reference signals
    Us_: ndarray
        Spatial filter
    Ts_: ndarray
        Spatial filter

    References
    ----------
    .. [1]  Zhang YU, Zhou G, Jin J, et al. Frequency recognition in SSVEP-based BCI using multiset canonical
        correlation analysis[J]. International journal of neural systems, 2014, 24(04): 1450013.


    """
    def __init__(
        self,
        n_components: int = 1,
        method: str = "msetcca2",
        n_jobs: Optional[int] = None,
    ):
        self.n_components = n_components
        self.method = method
        self.n_jobs = n_jobs

    def fit(self, X: ndarray, y: ndarray, Yf: Optional[ndarray] = None):
        """model train

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).
        y: ndarray
            Labels, shape(n_trials,)
        Yf: ndarray
            Reference signal(n_classes, 2*n_harmonics, n_samples)
        """
        if self.method == "msetcca2" and Yf is None:
            raise ValueError("The reference signals Yf should be provided.")
        self.classes_ = np.unique(y)
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        self.templates_ = np.stack(
            [np.mean(X[y == label], axis=0) for label in self.classes_]
        )

        if self.method == "msetcca2":
            Yf = cast(ndarray, Yf)
            Yf = np.reshape(Yf, (-1, *Yf.shape[-2:]))
            Yf = Yf - np.mean(Yf, axis=-1, keepdims=True)
            self.Yf_ = Yf
            feat = self.transform(X)
            self.clf_ = SVC()
            self.clf_.fit(feat, y)
        elif self.method == "msetcca1":
            self.Us_, self.Ts_ = zip(
                *[_msetcca_kernel1(X[y == label]) for label in self.classes_]
            )
            self.Us_, self.Ts_ = np.stack(self.Us_), np.stack(self.Ts_)
        return self

    def transform(self, X: ndarray):
        """Transform X into features and calculate the correlation coefficients of
        the signals from different trials.

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).

        Returns
        ----------
        rhos: ndarray
            The correlation coefficients, shape(n_trials, n_fre)
        """
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        n_components = self.n_components
        if self.method == "msetcca1":
            Ts = self.Ts_
            rhos = Parallel(n_jobs=self.n_jobs)(
                delayed(partial(_scca_feature, n_components=n_components))(a, Ts)
                for a in X
            )
            rhos = np.stack(rhos)
        elif self.method == "msetcca2":
            templates = self.templates_
            Yf = self.Yf_
            rhos = Parallel(n_jobs=self.n_jobs)(
                delayed(partial(_msetcca_feature2, n_components=n_components))(
                    a, templates, Yf
                )
                for a in X
            )
            rhos = np.stack(rhos)
        return rhos

    def predict(self, X: ndarray):
        """Predict the labels

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).

        Returns
        ----------
        labels: ndarray
            Predicting labels, shape(n_trials,).
        """
        feat = self.transform(X)
        if self.method == "msetcca1":
            labels = self.classes_[np.argmax(feat, axis=-1)]
        elif self.method == "msetcca2":
            labels = self.clf_.predict(feat)
        return labels


class FBMsetCCA(FilterBankSSVEP, ClassifierMixin):
    """
    The filter bank MsetCCA method, i.e., the MsetCCA method that combines the application of multiple filters in order
    to decompose the SSVEP signal into specific subbands[1]_.

    Parameters
    ----------
    filterbank: list[ndarray]
        Filter bank list.
    filterweights: ndarray
        Weights of filter banks.
    n_components: int
        The number of feature dimensions after dimensionality reduction, the dimension of the spatial filter,
        defaults to 1.
    n_jobs: int
        The number of CPU working cores, default is None.
    methods: str
        Two Pattern Feature Extraction and Fitting Classifier Model Methods Judgment, defaulting to 'msetcca2'.

    Attributes
    ----------
    classes_ : ndarray
        Predictive labels, obtained from labeled data by removing duplicate elements from it.

    References
    ----------
    .. [1] Zhang Y U, Zhou G, Jin J, et al. Frequency recognition in SSVEP-based BCI using multiset canonical
        correlation analysis[J]. International journal of neural systems, 2014, 24(04): 1450013.

    """
    def __init__(
        self,
        filterbank: List[ndarray],
        n_components: int = 1,
        method: str = "msetcca2",
        filterweights: Optional[ndarray] = None,
        n_jobs: Optional[int] = None,
    ):
        self.n_components = n_components
        self.method = method
        self.filterweights = filterweights
        self.n_jobs = n_jobs
        super().__init__(
            filterbank,
            MsetCCA(n_components=n_components, method=method),
            filterweights=filterweights,
            n_jobs=n_jobs,
        )

    def fit(self, X: ndarray, y: ndarray, Yf: Optional[ndarray] = None):  # type: ignore[override]
        """model train

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).
        y: ndarray
            Labels, shape(n_trials,).
        Yf: ndarray
            Reference signal(n_classes, 2*n_harmonics, n_samples).
        """
        self.classes_ = np.unique(y)
        super().fit(X, y, Yf=Yf)
        return self

    def predict(self, X: ndarray):
        """Predict the labels

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).

        Returns
        ----------
        labels: ndarray
            Predicting labels, shape(n_trials,).
        """
        features = self.transform(X)
        if self.filterweights is None:
            features = np.reshape(
                features, (features.shape[0], len(self.filterbank), -1)
            )
            features = np.mean(features, axis=1)
        labels = self.classes_[np.argmax(features, axis=-1)]
        return labels


def _msetccar_kernel(X: ndarray, Yf: ndarray):
    """Multi-set CCA1 with reference signals (MsetCCA-R).
    Parameters
    ----------
    X: ndarray
        EEG data, shape(n_trials, n_channels, n_samples).
    Yf: ndarray
        Reference signal, shape(n_classes, 2*n_harmonics, n_samples)


    """
    X = np.reshape(X, (-1, *X.shape[-2:]))
    M, C, N = X.shape
    n_components = min(C, Yf.shape[0])
    P = vstack([identity(N) for _ in range(M)])
    Q, R = qr(Yf.T, mode="economic")
    P = P @ Q @ Q.T @ P.T
    Z = block_diag(X).T
    _, U = _ged_wong(Z, None, P, n_components=n_components)  # U for X
    X = np.reshape(X, (-1, N))
    T = U.T @ X
    return U, T


class MsetCCAR(BaseEstimator, TransformerMixin, ClassifierMixin):

    def __init__(self, n_components: int = 1, n_jobs: Optional[int] = 1):
        self.n_components = n_components
        self.n_jobs = n_jobs

    def fit(self, X: ndarray, y: ndarray, Yf: ndarray):
        """model train

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).
        y: ndarray
            Labels, shape(n_trials,)
        Yf: ndarray
            Reference signal(n_classes, 2*n_harmonics, n_samples)
        """
        self.classes_ = np.unique(y)
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        self.templates_ = np.stack(
            [np.mean(X[y == label], axis=0) for label in self.classes_]
        )

        Yf = np.reshape(Yf, (-1, *Yf.shape[-2:]))
        Yf = Yf - np.mean(Yf, axis=-1, keepdims=True)
        self.Yf_ = Yf

        self.Us_, self.Ts_ = zip(
            *[
                _msetccar_kernel(X[y == label], self.Yf_[i])
                for i, label in enumerate(self.classes_)
            ]
        )
        # 每个类别的 Us_ shape 有的 (36, 9)，有的 (45, 9)，说明不同类别的 trial 数量（M）不一样。
        # 导致raise ValueError('all input arrays must have the same shape')
        # for i, (u, t) in enumerate(zip(self.Us_, self.Ts_)):
        #     print(f"Class {i}: Us_ shape = {np.shape(u)}, Ts_ shape = {np.shape(t)}")

        self.Us_, self.Ts_ = np.stack(self.Us_), np.stack(self.Ts_)
        return self

    def transform(self, X: ndarray):
        """Transform X into features and calculate the correlation coefficients of
        the signals from different trials

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).

        Returns
        ----------
        rhos: ndarray
            The correlation coefficients, shape(n_trials, n_fre)
        """
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        n_components = self.n_components

        Ts = self.Ts_
        rhos = Parallel(n_jobs=self.n_jobs)(
            delayed(partial(_scca_feature, n_components=n_components))(a, Ts) for a in X
        )
        rhos = np.stack(rhos)
        return rhos

    def predict(self, X: ndarray):
        """Predict the labels

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).

        Returns
        ----------
        labels: ndarray
            Predicting labels, shape(n_trials,).
        """
        feat = self.transform(X)
        labels = self.classes_[np.argmax(feat, axis=-1)]
        return labels


class FBMsetCCAR(FilterBankSSVEP, ClassifierMixin):

    def __init__(
        self,
        filterbank: List[ndarray],
        n_components: int = 1,
        filterweights: Optional[ndarray] = None,
        n_jobs: Optional[int] = None,
    ):
        self.n_components = n_components
        self.filterweights = filterweights
        self.n_jobs = n_jobs
        super().__init__(
            filterbank,
            MsetCCAR(n_components=n_components, n_jobs=1),
            filterweights=filterweights,
            n_jobs=n_jobs,
        )

    def fit(self, X: ndarray, y: ndarray, Yf: Optional[ndarray] = None):  # type: ignore[override]
        """model train

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).
        y: ndarray
            Labels, shape(n_trials,)
        Yf: ndarray
            Reference signal(n_classes, 2*n_harmonics, n_samples)
        """
        self.classes_ = np.unique(y)
        super().fit(X, y, Yf=Yf)
        return self

    def predict(self, X: ndarray):
        """Predict the labels

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).

        Returns
        ----------
        labels: ndarray
            Predicting labels, shape(n_trials,).
        """
        features = self.transform(X)
        if self.filterweights is None:
            features = np.reshape(
                features, (features.shape[0], len(self.filterbank), -1)
            )
            features = np.mean(features, axis=1)
        labels = self.classes_[np.argmax(features, axis=-1)]
        return labels


def _trca_kernel(X: ndarray):
    """TRCA spatial filter calculate.

    Parameters
    ----------
    X: (n_trials, n_channels, n_samples)
    """
    X = np.reshape(X, (-1, *X.shape[-2:]))
    M, C, N = X.shape
    n_components = C
    P = vstack([identity(N) for _ in range(M)])
    P = P @ P.T
    Z = np.hstack(X).T  # type: ignore
    _, U = _ged_wong(Z, None, P, n_components=n_components)  # U for X
    return U


def _trca_feature(
    X: ndarray,
    templates: ndarray,
    Us: ndarray,
    n_components: int = 1,
    ensemble: bool = True,
):
    rhos = []
    if not ensemble:
        for Xk, U in zip(templates, Us):
            a = U[:, :n_components].T @ X
            b = U[:, :n_components].T @ Xk
            a = np.reshape(a, (-1))
            b = np.reshape(b, (-1))
            rhos.append(pearsonr(a, b)[0])
    else:
        U = Us[:, :, :n_components]
        U = np.concatenate(U, axis=-1)
        for Xk in templates:
            a = U.T @ X
            b = U.T @ Xk
            a = np.reshape(a, (-1))
            b = np.reshape(b, (-1))
            rhos.append(pearsonr(a, b)[0])
    return rhos


class TRCA(BaseEstimator, TransformerMixin, ClassifierMixin):
    """The core idea of Task-Related Component Analysis (TRCA) algorithm is to extract task-related components by
    improving the repeatability between trials, specifically, the algorithm is based on inter-trial covariance matrix
    maximization to achieve the extraction of task-related components, which belongs to the supervised learning method[1]_.


    Parameters
    ----------
    n_components: int
        The number of feature dimensions after dimensionality reduction, the dimension of the spatial filter,
        defaults to 1.
    ensemble: bool
        Whether to perform spatial filter ensemble for each category of signals,
        the default is True to perform ensemble.
    n_jobs: int
        The number of CPU working cores, default is None.

    Attributes
    ----------
    classes_ : ndarray
        Predictive labels, obtained from labeled data by removing duplicate elements from it.
    templates_ : ndarray
        Individual average template
    Us_: ndarray
        Spatial filters obtained for each class of training signals.

    References
    ----------
    .. [1] Nakanishi M, Wang Y, Chen X, et al. Enhancing detection of SSVEPs for a high-speed brain speller using
        task-related component analysis. IEEE Transactions on Biomedical Engineering, 2018, 104-112.

    """
    def __init__(
        self, n_components: int = 1, ensemble: bool = True, n_jobs: Optional[int] = None
    ):
        self.n_components = n_components
        self.ensemble = ensemble
        self.n_jobs = n_jobs

    def fit(self, X: ndarray, y: ndarray, Yf: Optional[ndarray] = None):
        """model train

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).
        y: ndarray
            Labels, shape(n_trials,)
        Yf: ndarray
            Reference signal(n_classes, 2*n_harmonics, n_samples)
        """
        self.classes_ = np.unique(y)
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        self.templates_ = np.stack(
            [np.mean(X[y == label], axis=0) for label in self.classes_]
        )

        self.Us_ = np.stack([_trca_kernel(X[y == label]) for label in self.classes_])
        return self

    def transform(self, X: ndarray):
        """Transform X into features and calculate the correlation coefficients of
        the signals from different trials

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).

        Returns
        ----------
        rhos: ndarray
            The correlation coefficients, shape(n_trials, n_fre)
        """
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        n_components = self.n_components
        templates = self.templates_
        Us = self.Us_
        ensemble = self.ensemble
        rhos = Parallel(n_jobs=self.n_jobs)(
            delayed(
                partial(
                    _trca_feature, Us=Us, n_components=n_components, ensemble=ensemble
                )
            )(a, templates)
            for a in X
        )
        rhos = np.stack(rhos)
        return rhos

    def predict(self, X: ndarray):
        """Predict the labels

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).

        Returns
        ----------
        labels: ndarray
            Predicting labels, shape(n_trials,).
        """
        feat = self.transform(X)
        labels = self.classes_[np.argmax(feat, axis=-1)]
        return labels


class FBTRCA(FilterBankSSVEP, ClassifierMixin):
    """Filter bank TRCA (filter bank Task-Related Component Analysis, fbTRCA) adds the filter bank analysis method
    to TRCA by combining the fundamental and harmonic components of the signal. The EEG signal is first filtered using
    multiple subband filters with different cutoff frequencies to obtain the subband filtered signal. Subsequently,
    the correlation coefficients of the subband signals are summed according to a weighting function, and finally this
    weighted correlation coefficient sum is used as the feature discriminant [1]_.


    Parameters
    ----------
    filterbank: list[ndarray]
        Filter bank list
    filterweights: ndarray
        Filter weights, defaults to None.
    n_components: int
        The number of feature dimensions after dimensionality reduction, the dimension of the spatial filter,
        defaults to 1.
    n_jobs: int
        The number of CPU working cores, default is None.
    ensemble: bool
        Whether to perform spatial filter ensemble for each category of signals,
        the default is True to perform ensemble.

    Attributes
    ----------
    classes_ : ndarray
        Predictive labels, obtained from labeled data by removing duplicate elements from it.
    templates_ : ndarray
        Individual average template
    Us_: ndarray
        Spatial filters obtained for each class of training signals.


    References
    ----------
    .. [1] Nakanishi M, Wang Y, Chen X, et al. Enhancing detection of SSVEPs for a high-speed brain speller
        using task-related component analysis.IEEE Transactsions on Biomedical Engineering, 2018, 104-112.

    Tip
    ----
    .. code-block:: python
       :linenos:
       :emphasize-lines: 2
       :caption: A example using FBTRCA

        import numpy as np
        from brainda.algorithms.decomposition import FBTRCA
        X = np.zeros((4,22,22))
        for i in range(4):
            X[i,...] = np.identity(22)*0.5 + np.random.normal(-1,3,(22,22))*2
        y = np.array([1,1,2,2])
        filterbank = [np.ones((3,6))]
        filterweights = np.array([[0.3, -0.1], [0.5, -0.1]])
        estimator = FBTRCA(filterbank=filterbank,n_components=1, ensemble=True,filterweights=np.array(filterweights),n_jobs=-1)
        p_labels = estimator.fit(X, y)
        print(estimator.predict(np.identity(22)))
    """
    def __init__(
        self,
        filterbank: List[ndarray],
        n_components: int = 1,
        ensemble: bool = True,
        filterweights: Optional[ndarray] = None,
        n_jobs: Optional[int] = None,
    ):
        self.n_components = n_components
        self.ensemble = ensemble
        self.filterweights = filterweights
        self.n_jobs = n_jobs
        super().__init__(
            filterbank,
            TRCA(n_components=n_components, ensemble=ensemble, n_jobs=1),
            filterweights=filterweights,
            n_jobs=n_jobs,
        )

    def fit(self, X: ndarray, y: ndarray, Yf: Optional[ndarray] = None):  # type: ignore[override]
        """model train

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).
        y: ndarray
            Labels, shape(n_trials,)
        Yf: ndarray
            Reference signal, shape(n_classes, 2*n_harmonics, n_samples)
        """
        self.classes_ = np.unique(y)
        super().fit(X, y, Yf=Yf)
        return self

    def predict(self, X: ndarray):
        """Predict the labels

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).

        Returns
        ----------
        labels: ndarray
            Predicting labels, shape(n_trials,).
        """
        features = self.transform(X)
        if self.filterweights is None:
            features = np.reshape(
                features, (features.shape[0], len(self.filterbank), -1)
            )
            features = np.mean(features, axis=1)
        labels = self.classes_[np.argmax(features, axis=-1)]
        return labels


def _trcar_kernel(X: ndarray, Yf: ndarray):
    """TRCA-R.
    X: (n_trials, n_channels, n_samples)
    Yf: (n_harmonics, n_samples)
    """
    X = np.reshape(X, (-1, *X.shape[-2:]))
    M, C, N = X.shape
    n_components = C
    P = vstack([identity(N) for _ in range(M)])
    Q, R = qr(Yf.T, mode="economic")
    P = P @ Q @ Q.T @ P.T
    Z = np.hstack(X).T  # type: ignore
    _, U = _ged_wong(Z, None, P, n_components=n_components)  # U for X
    return U


class TRCAR(BaseEstimator, TransformerMixin, ClassifierMixin):
    """
    The task-related component analysis algorithm with sine-cosine reference signal (TRCA with sine-cosine reference
    signal, TRCA-R) is based on the TRCA algorithm, and the main improvement point is to add the step of orthogonal
    projection of the signal to the subspace of sine-cosine template during the training process, which further
    extracts the components of the EEG signal that are more correlated with the sine-cosine fluctuations of SSVEP[1]_.

    Parameters
    ----------
    n_components: int
        The number of feature dimensions after dimensionality reduction, the dimension of the spatial filter,
        defaults to 1.
    n_jobs: int
        The number of CPU working cores, default is None.
    ensemble: bool
        Whether to perform spatial filter ensemble for each category of signals,
        the default is True to perform ensemble.

    Attributes
    ----------
    classes_ : ndarray
        Predictive labels, obtained from labeled data by removing duplicate elements from it.
    templates_ : ndarray
        Individual average template.
    Yf_: ndarray
        Sine-Cosine reference signal.
    Us_: ndarray
        Spatial filters obtained for each class of training signals.


    References
    ----------
    .. [1] Wong C, Wang B, Wang Z, et al. Spatial Filtering in SSVEP-Based BCIs: Unified Framework and New
        Improvements. IEEE Transactions on Biomedical Engineering 2020, 3057-3072.

    Tip
    ----
    .. code-block:: python
       :linenos:
       :caption: A example using TRCAR

        import numpy as np
        from brainda.algorithms.decomposition import TRCAR
        X = np.array([[[0, -1],[2, -1]], [[2, -1],[0, 1]], [[1, -1],[3, 2]],[[-1, 2],[1, 0]]])
        y = np.array([1, 1, 2, 2])
        Yf = np.array([[[0, -0.5],[1, -1]], [[0.2, -1],[0, 1]]])
        estimator = TRCAR(n_components=1, ensemble=True, n_jobs=-1)
        p_labels = estimator.fit(X, y, Yf)
        print(estimator.predict(np.array([[[0, -1.2],[0.5, -1]]])))
    """
    def __init__(
        self, n_components: int = 1, ensemble: bool = True, n_jobs: Optional[int] = None
    ):
        self.n_components = n_components
        self.ensemble = ensemble
        self.n_jobs = n_jobs

    def fit(self, X: ndarray, y: ndarray, Yf: ndarray):
        """model train

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).
        y: ndarray
            Labels, shape(n_trials,)
        Yf: ndarray
            Reference signal(n_classes, 2*n_harmonics, n_samples)
        """
        self.classes_ = np.unique(y)
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        self.templates_ = np.stack(
            [np.mean(X[y == label], axis=0) for label in self.classes_]
        )

        Yf = np.reshape(Yf, (-1, *Yf.shape[-2:]))
        Yf = Yf - np.mean(Yf, axis=-1, keepdims=True)
        self.Yf_ = Yf

        self.Us_ = np.stack(
            [
                _trcar_kernel(X[y == label], self.Yf_[i])
                for i, label in enumerate(self.classes_)
            ]
        )
        return self

    def transform(self, X: ndarray):
        """Transform X into features and calculate the correlation coefficients of
        the signals from different trials

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).

        Returns
        ----------
        rhos: ndarray
            The correlation coefficients, shape(n_trials, n_fre)
        """
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        n_components = self.n_components
        templates = self.templates_
        Us = self.Us_
        ensemble = self.ensemble
        rhos = Parallel(n_jobs=self.n_jobs)(
            delayed(
                partial(
                    _trca_feature, Us=Us, n_components=n_components, ensemble=ensemble
                )
            )(a, templates)
            for a in X
        )
        rhos = np.stack(rhos)
        return rhos

    def predict(self, X: ndarray):
        """Predict the labels

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).

        Returns
        ----------
        labels: ndarray
            Predicting labels, shape(n_trials,).
        """
        feat = self.transform(X)
        labels = self.classes_[np.argmax(feat, axis=-1)]
        return labels


class FBTRCAR(FilterBankSSVEP, ClassifierMixin):
    """
    The filter bank TRCA-R algorithm (filter bank TRCA-R, fbTRCA-R) adds a filter bank analysis method to the TRCA-R
    algorithm, combining the fundamental and harmonic components of the signal. Multiple subband filters with different
    cutoff frequencies are utilized to filter the EEG signal to obtain the subband filtered signal. Subsequently,
    the correlation coefficients of the subband signals are summed according to a weighting function, and finally this
    weighted correlation coefficient sum is used as the feature discriminant[1]_.

    Parameters
    ----------
    filterbank: list[ndarray]
        Filter bank list
    filterweights: ndarray
        Filter weights, defaults to None.
    n_components: int
        The number of feature dimensions after dimensionality reduction, the dimension of the spatial filter,
        defaults to 1.
    n_jobs: int
        The number of CPU working cores, default is None.


    Attributes
    ----------
    classes_ : ndarray
        Predictive labels, obtained from labeled data by removing duplicate elements from it.
    ensemble: bool
        Whether to perform spatial filter ensemble for each category of signals,
        the default is True to perform ensemble.
    templates_ : ndarray
        Individual average template
    Us_: ndarray
        Spatial filters obtained for each class of training signals.
    Yf: ndarray
        Reference signal(n_classes, 2*n_harmonics, n_samples)


    References
    ----------
    .. [1] Chen X, Wang Y, Gao S, et al. Filter bank canonical correlation analysis for implementing a high-speed
       SSVEP-based brain-computer interface[J]. Journal of Neural Engineering, 2015, 12(4):046008.


    Tip
    ----
    .. code-block:: python
       :linenos:
       :emphasize-lines: 2
       :caption: A example using FBTRCAR

        import numpy as np
        from brainda.algorithms.decomposition import FBTRCAR
        X = np.zeros((4,22,22))
        for i in range(4):
        X[i,...] = np.identity(22)*0.3 + np.random.normal(-1,3,(22,22))*5
        y = np.array([1,1,2,2])
        Yf = X
        filterbank = [np.ones((3,6))]
        filterweights = np.array([[0.3, -0.1], [0.5, -0.1]])
        estimator = FBTRCAR(filterbank=filterbank,n_components=1,ensemble=True,filterweights=np.array(filterweights),n_jobs=-1)
        p_labels = estimator.fit(X, y, Yf)
        print(estimator.predict(np.identity(22)))

    """
    def __init__(
        self,
        filterbank: List[ndarray],
        n_components: int = 1,
        ensemble: bool = True,
        filterweights: Optional[ndarray] = None,
        n_jobs: Optional[int] = None,
    ):
        self.n_components = n_components
        self.ensemble = ensemble
        self.filterweights = filterweights
        self.n_jobs = n_jobs
        super().__init__(
            filterbank,
            TRCAR(n_components=n_components, ensemble=ensemble, n_jobs=1),
            filterweights=filterweights,
            n_jobs=n_jobs,
        )

    def fit(self, X: ndarray, y: ndarray, Yf: Optional[ndarray] = None):  # type: ignore[override]
        """model train

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).
        y: ndarray
            Labels, shape(n_trials,)
        Yf: ndarray
            Reference signal(n_classes, 2*n_harmonics, n_samples)
        """
        self.classes_ = np.unique(y)
        super().fit(X, y, Yf=Yf)
        return self

    def predict(self, X: ndarray):
        """Predict the labels

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_samples).

        Returns
        ----------
        labels: ndarray
            Predicting labels, shape(n_trials,).
        """
        features = self.transform(X)
        if self.filterweights is None:
            features = np.reshape(
                features, (features.shape[0], len(self.filterbank), -1)
            )
            features = np.mean(features, axis=1)
        labels = self.classes_[np.argmax(features, axis=-1)]
        return labels
