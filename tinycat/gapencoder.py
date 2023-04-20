from typing import Dict, Generator, List, Literal, Optional, Tuple, Union

from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.fixes import _object_dtype_isnan
from sklearn.utils import check_array
from sklearn.cluster import kmeans_plusplus
from sklearn.utils.extmath import row_norms
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.utils.extmath import row_norms, safe_sparse_dot
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_random_state, gen_batches

def check_input(X) -> np.ndarray:
    X_ = check_array(X, dtype=None, ensure_2d=True, force_all_finite=False)
    if X_.dtype.kind in {"U", "S"}:
        if np.any(X_ == "nan"):
            return check_array(
                np.array(X, dtype=object),
                dtype=None,
                ensure_2d=True,
                force_all_finite=False,
            )
    return X_

def batch_lookup(
    lookup: np.array,
    n: int = 1,
) -> Generator[Tuple[np.array, np.array], None, None]:
    len_iter = len(lookup)
    for idx in range(0, len_iter, n):
        indices = lookup[slice(idx, min(idx + n, len_iter))]
        unq_indices = np.unique(indices)
        yield unq_indices, indices


class GapEncoderColumn(BaseEstimator, TransformerMixin):
    rho_: float
    H_dict_: Dict[np.ndarray, np.ndarray]

    def __init__(
        self,
        n_components: int = 10,
        batch_size: int = 128,
        rho: float = 0.95,
        rescale_rho: bool = False,
        hashing: bool = False,
        analyzer: Literal["word", "char", "char_wb"] = "char",
        ngram_range: Tuple[int, int] = (2, 4),
        add_words: bool = False,
        init: Literal["k-means++", "random", "k-means"] = "k-means++",
        tol: float = 1e-4,
        min_iter: int = 2,
        max_iter: int = 5,
        max_iter_e_step: int = 20,
        rescale_W: bool = True,
        gamma_shape_prior: float = 1.1,
        gamma_scale_prior: float = 1.0,
    ):
        self.n_components = n_components
        self.rho = rho
        self.rescale_rho = rescale_rho
        self.hashing = hashing
        self.analyzer = analyzer
        self.ngram_range = ngram_range
        self.add_words = add_words
        self.init = init
        self.batch_size = batch_size
        self.tol = tol
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.max_iter_e_step = max_iter_e_step
        self.rescale_W = rescale_W
        self.gamma_shape_prior = gamma_shape_prior
        self.gamma_scale_prior = gamma_scale_prior

    def _init_vars(self, X) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.hashing:
            self.ngrams_count_ = HashingVectorizer(
                analyzer=self.analyzer,
                ngram_range=self.ngram_range,
                n_features=self.hashing_n_features,
                norm=None,
                alternate_sign=False
            )
            if self.add_words:
                self.word_count_ = HashingVectorizer(
                    analyzer='word',
                    n_features=self.hashing_n_features,
                    norm=None,
                    alternate_sign=False
                )
        else:
            self.ngrams_count_ = CountVectorizer(
                analyzer=self.analyzer, ngram_range=self.ngram_range, dtype=np.float64
            )
            if self.add_words:
                self.word_count_ = CountVectorizer(dtype=np.float64)

        self.H_dict_ = dict()
        unq_X, lookup = np.unique(X, return_inverse=True)
        unq_V = self.ngrams_count_.fit_transform(unq_X)
        if self.add_words:
            unq_V2 = self.word_count_.fit_transform(unq_X)
            unq_V = sparse.hstack((unq_V, unq_v2), format="csr")

        if not self.hashing:
            self.vocabulary = self.ngrams_count_.get_feature_names_out()

        if self.add_words:
            self.vocabulary = np.concatenate(
                (self.vocabulary, self.word_count_.get_feature_names_out())
            )

        _, self.n_vocab = unq_V.shape
        self.W_, self.A_, self.B_ = self._init_w(unq_V[lookup], X)
        unq_H = _rescale_h(unq_V, np.ones((len(unq_X), self.n_components)))
        self.H_dict_.update(zip(unq_X, unq_H))
        if self.rescale_rho:
            self.rho_ = self.rho ** (self.batch_size / len(X))
        return unq_X, unq_V, lookup
        
    def _init_w(self, V: np.array, X) -> Tuple[np.array, np.array, np.array]:
        if self.init == "k-means++":
            W, _ = kmeans_plusplus(V, self.n_components,
                x_squared_norms=row_norms(V, squared=True),
                n_local_trials=None,
            )
        W /= W.sum(axis=1, keepdims=True)
        A = np.ones((self.n_components, self.n_vocab)) * 1e-10
        B = A.copy()
        return W, A, B

    def _get_H(self, X:np.array) -> np.array:
        H_out = np.empty((len(X), self.n_components))
        for x, h_out in zip(X, H_out):
            h_out[:] = self.H_dict_[x]
        return H_out

    def _add_unseen_keys_to_H_dict(self, X) -> None:
        unseen_X = np.setdiff1d(X, np.array([*self.H_dict_]))
        if unseen_X.size > 0:
            unseen_V = self.ngrams_count_.transform(unseen_X)
            if self.add_words:
                unseen_V2 = self.word_count_.transform(unseen_X)
                unseen_V = sparse.hstack((unseen_V, unseen_V2), format="csr")

            unseen_H = _rescale_h(
                unseen_V, np.ones((unseen_V.shape[0], self.n_components))
            )
            self.H_dict_.update(zip(unseen_X, unseen_H))

    def fit(self, X, y=None) -> "GapEncoderColumn":
        self.rho_ = self.rho
        assert isinstance(X[0], str), "Input data is not string"

        # TODO::: check cudf/gpu compat
        unq_X, unq_V, lookup = self._init_vars(X)
        n_batch = (len(X) - 1) // self.batch_size + 1
        del X
        
        # TODO::: check cudf/gpu compat
        unq_H = self._get_H(unq_X)

        for n_iter_ in range(self.max_iter):
            for i, (unq_idx, idx) in enumerate(batch_lookup(lookup, n=self.batch_size)):
                if i == n_batch - 1:
                    W_last = self.W_.copy()
                unq_H[unq_idx] = _multiplicative_update_h(
                    unq_V[unq_idx],
                    self.W_,
                    unq_H[unq_idx],
                    epsilon=1e-3,
                    max_iter=self.max_iter_e_step,
                    rescale_W=self.rescale_W,
                    gamma_shape_prior=self.gamma_shape_prior,
                    gamma_scale_prior=self.gamma_scale_prior,
                )
                _multiplicative_update_w(
                    unq_V[idx],
                    self.W_,
                    self.A_,
                    self.B_,
                    unq_H[idx],
                    self.rescale_W,
                    self.rho_,
                )
                if i == n_batch - 1:
                    # TODO:: cudf ops
                    W_change = np.linalg.norm(self.W_ - W_last) / np.linalg.norm(W_last)

            if (W_change < self.tol) and (n_iter_ >= self.min_iter - 1):
                break

        self.H_dict_.update(zip(unq_X, unq_H))
        return self

    def get_feature_names_out(
        self,
        n_labels: int = 3,
        prefix: str = "",
    ) -> List[str]:

        vectorizer = CountVectorizer()
        vectorizer.fit(list(self.H_dict_.keys()))
        vocabulary = np.array(vectorizer.get_feature_names_out())
        encoding = self.transform(np.array(vocabulary).reshape(-1))
        encoding = abs(encoding)
        encoding = encoding / np.sum(encoding, axis=1, keepdims=True)
        n_components = encoding.shape[1]
        topic_labels = []
        for i in range(n_components):
            x = encoding[:, i]
            labels = vocabulary[np.argsort(-x)[:n_labels]]
            topic_labels.append(labels)
        topic_labels = [prefix + ", ".join(label) for label in topic_labels]
        return topic_labels

    def transform(self, X) -> np.array:
        check_is_fitted(self, "H_dict_")
        pre_trans_H_dict_ = deepcopy(self.H_dict_)
        assert isinstance(X[0], str), "Input data is not string. "
        unq_X = np.unique(X)
        unq_V = self.ngrams_count_.transform(unq_X)
        if self.add_words:  # Add words counts
            unq_V2 = self.word_count_.transform(unq_X)
            unq_V = sparse.hstack((unq_V, unq_V2), format="csr")
        self._add_unseen_keys_to_H_dict(unq_X)
        unq_H = self._get_H(unq_X)
        for slc in gen_batches(n=unq_H.shape[0], batch_size=self.batch_size):
            unq_H[slc] = _multiplicative_update_h(
                unq_V[slc],
                self.W_,
                unq_H[slc],
                epsilon=1e-3,
                max_iter=100,
                rescale_W=self.rescale_W,
                gamma_shape_prior=self.gamma_shape_prior,
                gamma_scale_prior=self.gamma_scale_prior,
            )
        self.H_dict_.update(zip(unq_X, unq_H))
        feature_names_out = self._get_H(X)
        self.H_dict_ = pre_trans_H_dict_
        return feature_names_out

class GapEncoder(BaseEstimator, TransformerMixin):
    rho_: float
    fitted_models_: List[GapEncoderColumn]
    column_names_ : List[str]

    def __init__(
        self,
        n_components: int = 10,
        batch_size: int = 128,
        rho: float = 0.95,
        rescale_rho: bool = False,
        handle_missing: Literal["error", "empty_impute"] = "zero_impute",
        hashing: bool = False,
        analyzer: Literal["word", "char", "char_wb"] = "char",
        ngram_range: Tuple[int, int] = (2, 4),
        add_words: bool = False,
        init: Literal["k-means++", "random", "k-means"] = "k-means++",
        tol: float = 1e-4,
        min_iter: int = 2,
        max_iter: int = 5,
        max_iter_e_step: int = 20,
        rescale_W: bool = True,
        gamma_shape_prior: float = 1.1,
        gamma_scale_prior: float = 1.0,
    ):
        self.n_components = n_components
        self.batch_size = batch_size
        self.rho = rho
        self.rescale_rho = rescale_rho
        self.handle_missing = handle_missing
        self.hashing = hashing
        self.analyzer = analyzer
        self.ngram_range = ngram_range
        self.add_words = add_words
        self.init = init
        self.tol = tol
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.max_iter_e_step = max_iter_e_step
        self.rescale_W = rescale_W
        self.gamma_shape_prior = gamma_shape_prior
        self.gamma_scale_prior = gamma_scale_prior

    def _more_tags(self) -> Dict[str, List[str]]:
        return {"X_types": ["cateforical"]}

    def _create_column_gap_encoder(self) -> GapEncoderColumn:
        return GapEncoderColumn(
            n_components=self.n_components,
            batch_size=self.batch_size,
            rho=self.rho,
            rescale_rho=self.rescale_rho,
            hashing=self.hashing,
            analyzer=self.analyzer,
            ngram_range=self.ngram_range,
            add_words=self.add_words,
            tol=self.tol,
            min_iter=self.min_iter,
            max_iter=self.max_iter,
            max_iter_e_step=self.max_iter_e_step,
            rescale_W=self.rescale_W,
            gamma_shape_prior=self.gamma_shape_prior,
            gamma_scale_prior=self.gamma_scale_prior,
        )

    def _handle_missing(self, X):
        if self.handle_missing not in ['error', 'zero_impute']:
            raise ValueError(
                "handle_missing should be either 'error' or "
                f"'zero_impute', got {self.handle_missing!r}. "
            )

        # does _object_dtype_isnan works with cudf dataframe?
        missing_mask = _object_dtype_isnan(X)

        if missing_mask.any():
            if self.handle_missing == "error":
                raise ValueError("Input data contains missing values")
            elif self.handle_missing == "zero_impute":
                X[missing_mask] = " "
        return X

    def fit(self, X, y=None) -> "GapEncoder":
        if len(X) < self.n_components:
            raise ValueError(
                f"n_samples={len(X)} should be >= n_components={self.n_components}. "
            )
        self.rho_ = self.rho

        if isinstance(X, pd.DataFrame):
            # TODO: check if X.columns works for cudf
            self.column_names_ = list(X.columns)
        
        # TODO: need to make sure if it works for cudf
        X = check_input(X)
        X = self._handle_missing(X)
        self.fitted_models_ = []

        for k in range(X.shape[1]):
            col_enc = self._create_column_gap_encoder()
            self.fitted_models_.append(col_enc.fit(X[:, k]))
        return self

    def get_feature_names_out(
        self,
        col_names: Optional[Union[Literal["auto"], List[str]]] = None,
        n_labels: int = 3,
    ):
        check_is_fitted(self, "fitted_models_")
        if isinstance(col_names, str) and col_names == "auto":
            if hasattr(self, "column_names_"):  # Use column names
                prefixes = ["%s: " % col for col in self.column_names_]
            else:  # Use 'col1: ', ... 'colN: ' as prefixes
                prefixes = ["col%d: " % i for i in range(len(self.fitted_models_))]
        elif col_names is None:  # Empty prefixes
            prefixes = [""] * len(self.fitted_models_)
        else:
            prefixes = ["%s: " % col for col in col_names]
        labels = list()
        for k, enc in enumerate(self.fitted_models_):
            col_labels = enc.get_feature_names_out(n_labels, prefixes[k])
            labels.extend(col_labels)
        return labels

        
def _multiplicative_update_h(
    Vt: np.array,
    W: np.array,
    Ht: np.array,
    epsilon: float = 1e-3,
    max_iter: int = 10,
    rescale_W: bool = False,
    gamma_shape_prior: float = 1.1,
    gamma_scale_prior: float = 1.0,
):
    if rescale_W:
        WT1 = 1 + 1 / gamma_scale_prior
        W_WT1 = W / WT1
    else:
        WT1 = np.sum(W, axis=1) + 1 / gamma_scale_prior
        W_WT1 = W / WT1.reshape(-1, 1)
    
    const = (gamma_shape_prior - 1) / WT1
    squared_epsilon = epsilon**2
    for vt, ht in zip(Vt, Ht):
        vt_ = vt.data
        idx = vt.indices
        W_WT1_ = W_WT1[:, idx]
        W_ = W[:, idx]
        squared_norm = 1
        for n_iter_ in range(max_iter):
            if squared_norm <= squared_epsilon:
                break
            aux = np.dot(W_WT1_, vt_ / (np.dot(ht, W_) + 1e-10))
            ht_out = ht * aux + const
            squared_norm = np.dot(ht_out - ht, ht_out - ht) / np.dot(ht, ht)
            ht[:] = ht_out
    return Ht


def _multiplicative_update_w(
    Vt: np.array,
    W: np.array,
    A: np.array,
    B: np.array,
    Ht: np.array,
    rescale_W: bool,
    rho: float,
) -> Tuple[np.array, np.array, np.array]:
    A *= rho
    A += W * safe_sparse_dot(Ht.T, Vt.multiply(1 / (np.dot(Ht, W) + 1e-10)))
    B *= rho
    B += Ht.sum(axis=0).reshape(-1, 1)
    np.divide(A, B, out=W)
    if rescale_W:
        _rescale_W(W, A)
    return W, A, B

def _rescale_W(W: np.array, A: np.array) -> None:
    s = W.sum(axis=1, keepdims=True)
    W /= s
    A /= s

def _rescale_h(V: np.array, H: np.array) -> np.array:
    epsilon = 1e-10
    H *= np.maximum(epsilon, V.sum(axis=1).A)
    H /= H.sum(axis=1, keepdims=True)
    return H
