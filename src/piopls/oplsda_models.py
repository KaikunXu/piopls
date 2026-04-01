# src/piopls/oplsda_models.py
"""Orthogonal Partial Least Squares Discriminant Analysis (OPLS-DA) module.

This module provides the Python implementation of OPLS-DA, strictly aligned 
with the R 'ropls' package. It calculates step-wise sequential increments 
for R2X, R2Y, and Q2, and features parallel processing for permutation tests.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.preprocessing import LabelEncoder
from joblib import Parallel, delayed
from tqdm.auto import tqdm


class OPLSDA:
    """Python implementation of OPLS-DA strictly aligned with R 'ropls'.

    Calculates step-wise sequential increments for R2X, R2Y, and Q2 to 
    perfectly match the 'ropls' model overview definitions. Includes methods 
    to export model parameters and plotting data as pandas DataFrames. 

    Attributes:
        n_ortho (int): Number of orthogonal components.
        cv_folds (int): Number of cross-validation folds.
        max_ortho (int): Maximum orthogonal components for auto-search.
        n_perms (int): Default number of permutations for validation.
        n_jobs (int): Default number of CPU cores for parallel jobs.
        label_encoder (LabelEncoder): Encoder for categorical targets.
        _is_categorical (bool): Flag for categorical target variable.
    """

    def __init__(
        self, 
        n_ortho=None, 
        cv_folds=7, 
        max_ortho=10, 
        n_perms=100, 
        n_jobs=-1
    ):
        """Initializes the OPLSDA model with specific configurations.

        Args:
            n_ortho (int, optional): Number of orthogonal components.
                If None, it will be automatically determined. Defaults to None.
            cv_folds (int, optional): Folds for cross-validation. Defaults to 7.
            max_ortho (int, optional): Max components to search. Defaults to 10.
            n_perms (int, optional): Permutations for test. Defaults to 100.
            n_jobs (int, optional): Cores for parallel compute. Defaults to -1.
        """
        self.n_ortho = n_ortho       
        self.cv_folds = cv_folds     
        self.max_ortho = max_ortho
        self.n_perms = n_perms
        self.n_jobs = n_jobs
        self.label_encoder = LabelEncoder()
        self._is_categorical = False 

    def _venetian_blinds_split(self, y_numeric):
        """Splits data using the Venetian blinds cross-validation strategy.

        Args:
            y_numeric (np.ndarray): The numeric target array.

        Returns:
            list of tuples: A list containing (train_indices, test_indices) 
                for each cross-validation fold.
        """
        folds = [([], []) for _ in range(self.cv_folds)]
        for class_val in np.unique(y_numeric):
            idx_c = np.where(y_numeric == class_val)[0]
            for i in range(self.cv_folds):
                test_c = idx_c[i::self.cv_folds]
                train_c = np.setdiff1d(idx_c, test_c)
                folds[i][0].extend(train_c)
                folds[i][1].extend(test_c)
        return [(np.array(train), np.array(test)) for train, test in folds]

    def _find_best_n_ortho(self, X, y_numeric):
        """Automatically finds optimal number of orthogonal components.

        Evaluates components sequentially and stops when the Q2 increment 
        falls below the defined threshold (0.01).

        Args:
            X (np.ndarray): The feature matrix.
            y_numeric (np.ndarray): The numeric target array.

        Returns:
            int: The optimal number of orthogonal components.
        """
        base_model = OPLSDA(n_ortho=0, cv_folds=self.cv_folds)
        base_model.fit(X, y_numeric)
        base_model.compute_q2(X, y_numeric)
        best_n = 0
        q2_prev = base_model.Q2_
        
        for n in range(1, self.max_ortho + 1):
            temp_model = OPLSDA(n_ortho=n, cv_folds=self.cv_folds)
            temp_model.fit(X, y_numeric)
            temp_model.compute_q2(X, y_numeric)
            q2_curr = temp_model.Q2_
            
            # Threshold improvement of 0.01 for Q2 to accept new component
            if q2_curr - q2_prev >= 0.01:
                best_n = n
                q2_prev = q2_curr
            else:
                break
        return max(1, best_n)

    def fit(self, X, y):
        """Fits the OPLS-DA model to the provided data.

        Args:
            X (array-like): Feature matrix of shape (n_samples, n_features).
            y (array-like): Target vector of shape (n_samples,).

        Returns:
            self: The fitted OPLSDA object.
        """
        # Extract metadata if available (e.g., from pandas DataFrames)
        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns.tolist()
        if hasattr(X, 'index'):
            self.sample_names_in_ = X.index.tolist()
            
        X = np.asarray(X, dtype=float)
        y_array = np.asarray(y)
        
        # Determine if target is categorical and encode it
        if y_array.dtype.kind in {'U', 'S', 'O'}: 
            y_numeric = self.label_encoder.fit_transform(y_array).astype(float)
            self._is_categorical = True
        else:
            y_numeric = y_array.astype(float)
            self._is_categorical = False
            self.label_encoder.fit(y_numeric) 
            
        n_samples, n_features = X.shape
        if self.n_ortho is None:
            self.n_ortho = self._find_best_n_ortho(X, y_numeric)

        # Scale X matrix (Unit Variance / Z-score scaling)
        self.x_mean_ = X.mean(axis=0)
        self.x_std_ = X.std(axis=0, ddof=1)
        self.x_std_[self.x_std_ == 0] = 1.0  # Prevent division by zero
        E = (X - self.x_mean_) / self.x_std_
        E_orig = E.copy() 

        # Scale Y vector
        self.y_mean_ = y_numeric.mean()
        self.y_std_ = y_numeric.std(ddof=1)
        if self.y_std_ == 0: 
            self.y_std_ = 1.0
        f = (y_numeric - self.y_mean_) / self.y_std_

        # Calculate Total Sum of Squares for normalization
        SS_X_total = np.sum(E_orig ** 2)
        SS_Y_total = np.sum((y_numeric - self.y_mean_) ** 2)

        _n_ortho = max(0, self.n_ortho)
        
        # Initialize storage matrices for orthogonal components
        self.T_ortho_ = np.zeros((n_samples, _n_ortho)) 
        self.P_ortho_ = np.zeros((n_features, _n_ortho)) 
        self.W_ortho_ = np.zeros((n_features, _n_ortho))
        
        self.R2X_cum_list_ = []
        self.R2Y_cum_list_ = []

        # ---- Step 0: Initial Predictive Component ----
        w_pred_0 = np.dot(E.T, f)
        w_pred_0 /= np.linalg.norm(w_pred_0)
        
        t_pred_0 = np.dot(E, w_pred_0)
        p_pred_0 = np.dot(E.T, t_pred_0) / np.dot(t_pred_0.T, t_pred_0)
        c_pred_0 = np.dot(f.T, t_pred_0) / np.dot(t_pred_0.T, t_pred_0)
        
        R2X_cum_0 = (np.sum(t_pred_0**2) * np.sum(p_pred_0**2)) / SS_X_total
        y_pred_0 = (t_pred_0 * c_pred_0) * self.y_std_ + self.y_mean_
        R2Y_cum_0 = 1.0 - np.sum((y_numeric - y_pred_0) ** 2) / SS_Y_total
        
        self.R2X_cum_list_.append(R2X_cum_0)
        self.R2Y_cum_list_.append(R2Y_cum_0)
        
        if _n_ortho == 0:
            self.w_pred_ = w_pred_0
            self.t_pred_ = t_pred_0
            self.p_pred_ = p_pred_0
            self.c_pred_ = c_pred_0
            self.R2X_ = R2X_cum_0
            self.R2Y_ = R2Y_cum_0
            self.RMSEE_ = np.sqrt(np.mean((y_numeric - y_pred_0) ** 2))

        # ---- Step 1 to n: Extract Orthogonal Components ----
        for i in range(_n_ortho):
            # NIPALS iterative approach
            w = np.dot(E.T, f)
            w /= np.linalg.norm(w)
            t = np.dot(E, w)
            p = np.dot(E.T, t) / np.dot(t.T, t)
            
            # Orthogonal Signal Correction (OSC) filtering
            w_ortho = p - np.dot(w.T, p) * w
            w_ortho /= np.linalg.norm(w_ortho)
            t_ortho = np.dot(E, w_ortho)
            
            # Gram-Schmidt orthogonalization
            for j in range(i):
                t_prev = self.T_ortho_[:, j]
                proj = np.dot(t_ortho.T, t_prev) / np.dot(t_prev.T, t_prev)
                t_ortho -= proj * t_prev
                
            p_ortho = np.dot(E.T, t_ortho) / np.dot(t_ortho.T, t_ortho)            
            
            self.T_ortho_[:, i] = t_ortho
            self.P_ortho_[:, i] = p_ortho
            self.W_ortho_[:, i] = w_ortho
            
            # Deflate the X matrix
            E = E - np.outer(t_ortho, p_ortho)

            # Recalculate predictive components on deflated E
            w_pred_k = np.dot(E.T, f)
            w_pred_k /= np.linalg.norm(w_pred_k)
            t_pred_k = np.dot(E, w_pred_k)
            
            # Ensure predictive scores are orthogonal
            for j in range(i + 1):
                t_prev = self.T_ortho_[:, j]
                proj = np.dot(t_pred_k.T, t_prev) / np.dot(t_prev.T, t_prev)
                t_pred_k -= proj * t_prev

            p_pred_k = np.dot(E.T, t_pred_k) / np.dot(t_pred_k.T, t_pred_k)
            c_pred_k = np.dot(f.T, t_pred_k) / np.dot(t_pred_k.T, t_pred_k)
            
            # Update cumulative variances
            ss_ortho = (
                np.sum(self.T_ortho_[:, :i+1]**2, axis=0) * np.sum(
                    self.P_ortho_[:, :i+1]**2, axis=0)
            )
            ss_pred = np.sum(t_pred_k**2) * np.sum(p_pred_k**2)
            R2X_cum_k = (np.sum(ss_ortho) + ss_pred) / SS_X_total
            
            y_pred_k = (t_pred_k * c_pred_k) * self.y_std_ + self.y_mean_
            R2Y_cum_k = 1.0 - np.sum((y_numeric - y_pred_k) ** 2) / SS_Y_total
            
            self.R2X_cum_list_.append(R2X_cum_k)
            self.R2Y_cum_list_.append(R2Y_cum_k)
            
            if i == _n_ortho - 1:
                self.w_pred_ = w_pred_k
                self.t_pred_ = t_pred_k
                self.p_pred_ = p_pred_k
                self.c_pred_ = c_pred_k
                self.R2X_ = R2X_cum_k
                self.R2Y_ = R2Y_cum_k
                self.RMSEE_ = np.sqrt(np.mean((y_numeric - y_pred_k) ** 2))

        # ---- Transform cumulative metrics into sequential increments ----
        self.R2X_comp_ = []
        self.R2Y_comp_ = []
        for i in range(len(self.R2X_cum_list_)):
            if i == 0:
                self.R2X_comp_.append(self.R2X_cum_list_[i])
                self.R2Y_comp_.append(self.R2Y_cum_list_[i])
            else:
                inc_x = self.R2X_cum_list_[i] - self.R2X_cum_list_[i-1]
                inc_y = self.R2Y_cum_list_[i] - self.R2Y_cum_list_[i-1]
                self.R2X_comp_.append(inc_x)
                self.R2Y_comp_.append(inc_y)

        # Calculate Variable Importance in Projection (VIP)
        if _n_ortho > 0:
            W_all = np.column_stack((self.w_pred_, self.W_ortho_))
            P_all = np.column_stack((self.p_pred_, self.P_ortho_))
            W_star = np.dot(W_all, np.linalg.inv(np.dot(P_all.T, W_all)))
            w_star_pred = W_star[:, 0]
        else:
            w_star_pred = self.w_pred_
            
        w_star_pred /= np.linalg.norm(w_star_pred)
        self.vip_ropls_ = np.sqrt(n_features * (w_star_pred ** 2))

        # Dynamically calculate Covariance/Correlation for S-Plot
        t_pred_flat = self.t_pred_.flatten()
        self.covariances_ = np.zeros(n_features)
        self.correlations_ = np.zeros(n_features)
        for i in range(n_features):
            xi = X[:, i]
            self.covariances_[i] = np.cov(xi, t_pred_flat)[0, 1]
            self.correlations_[i] = stats.pearsonr(xi, t_pred_flat)[0]

        return self

    def _predict_continuous(self, X):
        """Internal method to predict continuous target values.

        Args:
            X (np.ndarray): Feature matrix to predict.

        Returns:
            np.ndarray: Predicted continuous values.
        """
        X = np.asarray(X, dtype=float)
        E = (X - self.x_mean_) / self.x_std_
        
        for i in range(self.n_ortho):
            t_ortho = np.dot(E, self.W_ortho_[:, i])
            E -= np.outer(t_ortho, self.P_ortho_[:, i])
            
        t_pred = np.dot(E, self.w_pred_)
        y_pred_num = (t_pred * self.c_pred_) * self.y_std_ + self.y_mean_
        return y_pred_num

    def predict(self, X):
        """Predicts class labels or continuous values for the input data.

        Args:
            X (array-like): Feature matrix.

        Returns:
            array-like: Predicted class labels or continuous values.
        """
        y_pred_num = self._predict_continuous(X)
        if self._is_categorical:
            n_classes = len(self.label_encoder.classes_)
            y_pred_idx = np.clip(
                np.round(y_pred_num).astype(int), 0, n_classes - 1
            )
            return self.label_encoder.inverse_transform(y_pred_idx)
        return y_pred_num

    def compute_q2(self, X, y):
        """Calculates the Q2 cross-validated predictive ability.

        Args:
            X (array-like): Feature matrix.
            y (array-like): Target vector.

        Returns:
            float: The global Q2 score of the model.
        """
        X = np.asarray(X, dtype=float)
        if self._is_categorical:
            y_num = self.label_encoder.transform(y).astype(float)
        else:
            y_num = np.asarray(y, dtype=float)
            
        n_ortho_plus_1 = self.n_ortho + 1
        y_cv_preds = {n: np.zeros_like(y_num) for n in range(n_ortho_plus_1)}
        folds = self._venetian_blinds_split(y_num)
        
        for train_idx, test_idx in folds:
            X_train, y_train = X[train_idx], y_num[train_idx]
            X_test = X[test_idx]
            
            for n in range(n_ortho_plus_1):
                model_cv = OPLSDA(n_ortho=n, cv_folds=self.cv_folds)
                model_cv.fit(X_train, y_train)
                y_cv_preds[n][test_idx] = model_cv._predict_continuous(X_test)
                
        SS_Y = np.sum((y_num - y_num.mean()) ** 2)
        PRESS_list = [
            np.sum((y_num - y_cv_preds[n]) ** 2) 
            for n in range(n_ortho_plus_1)
        ]
        
        self.Q2_comp_ = []
        for i in range(len(PRESS_list)):
            if i == 0:
                self.Q2_comp_.append(1.0 - PRESS_list[i] / SS_Y)
            else:
                prev_press = PRESS_list[i-1] if PRESS_list[i-1] != 0 else 1e-10
                self.Q2_comp_.append(1.0 - PRESS_list[i] / prev_press)
                
        self.Q2_ = 1.0 - (PRESS_list[-1] / SS_Y)
        return self.Q2_

    def _single_permutation(self, X, y_numeric, n_ortho, cv_folds):
        """Runs a single iteration of the permutation test.

        Args:
            X (np.ndarray): Feature matrix.
            y_numeric (np.ndarray): Original numeric target vector.
            n_ortho (int): Number of orthogonal components.
            cv_folds (int): Number of cross-validation folds.

        Returns:
            tuple: Permuted (R2Y, Q2) metrics.
        """
        y_perm = np.random.permutation(y_numeric) 
        model_perm = OPLSDA(n_ortho=n_ortho, cv_folds=cv_folds)
        model_perm.fit(X, y_perm)
        q2 = model_perm.compute_q2(X, y_perm)
        return model_perm.R2Y_, q2

    def permutation_test(self, X, y, n_perms=None, n_jobs=None):
        """Performs a permutation test to assess model significance.
        
        Falls back to class attributes n_perms and n_jobs if not specified.

        Args:
            X (array-like): Feature matrix.
            y (array-like): Target vector.
            n_perms (int, optional): Number of permutations. Defaults to None.
            n_jobs (int, optional): Cores for parallel jobs. Defaults to None.

        Returns:
            dict: Dictionary containing real and permuted values of R2Y and Q2, 
                and empirical p-values.
        """
        _perms = n_perms if n_perms is not None else self.n_perms
        _jobs = n_jobs if n_jobs is not None else self.n_jobs
        
        if not hasattr(self, 'Q2_'):
            self.compute_q2(X, y)
            
        real_R2Y, real_Q2 = self.R2Y_, self.Q2_
        
        if self._is_categorical:
            y_num = self.label_encoder.transform(y).astype(float)
        else:
            y_num = np.asarray(y, dtype=float)
            
        # Parallel Task Generator
        tasks = (
            delayed(self._single_permutation)(
                X, y_num, self.n_ortho, self.cv_folds
            )
            for _ in range(_perms)
        )
        
        # Performs parallel computation of permutation checks, and displays 
        # a progress bar using tqdm
        try:
            # Accurate completion tracking using the generator mode (>=1.3)
            results = list(tqdm(
                Parallel(n_jobs=_jobs, return_as="generator")(tasks),
                total=_perms,
                desc="Permutation Test",
                leave=True,
                colour="#7F7F7F",
                bar_format="{l_bar}{bar:80}{r_bar}"
            ))
        except TypeError:
            # Older versions: task-based delivery progress tracking
            results = Parallel(n_jobs=_jobs)(
                delayed(self._single_permutation)(
                    X, y_num, self.n_ortho, self.cv_folds
                )
                for _ in tqdm(
                    range(_perms),
                    desc="Permutation Test",
                    leave=True,
                    colour="#7F7F7F",
                    bar_format="{l_bar}{bar:80}{r_bar}"
                )
            )
        
        perms_R2Y = [res[0] for res in results]
        perms_Q2 = [res[1] for res in results]
            
        p_R2Y = (np.sum(np.array(perms_R2Y) >= real_R2Y) + 1) / (_perms + 1)
        p_Q2 = (np.sum(np.array(perms_Q2) >= real_Q2) + 1) / (_perms + 1)
        
        return {
            'R2Y_real': real_R2Y, 'Q2_real': real_Q2,
            'p_R2Y': p_R2Y, 'p_Q2': p_Q2,
            'perms_R2Y': perms_R2Y, 'perms_Q2': perms_Q2
        }

    # ==========================================
    # DataFrame Export Methods
    # ==========================================
    def get_summary_df(self):
        """Exports the model overview metrics as a DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing R2X, R2Y, and Q2 metrics 
                for each extracted component.
        """
        if not hasattr(self, 'Q2_comp_'):
            raise ValueError("Model must compute Q2 first.")
            
        n_ortho = getattr(self, 'n_ortho', 0)
        components = ['p1'] + [f'o{i+1}' for i in range(n_ortho)]
        
        return pd.DataFrame({
            'Component': components, 
            'R2X': self.R2X_comp_, 
            'R2Y': self.R2Y_comp_, 
            'Q2': self.Q2_comp_
        })

    def get_scores_df(self, sample_names=None, y_true=None):
        """Exports the predictive and orthogonal scores as a DataFrame.

        Args:
            sample_names (list, optional): Sample names. Defaults to None.
            y_true (array-like, optional): True class labels. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame containing T scores.
        """
        n_samples = self.t_pred_.shape[0]
        if sample_names is not None: 
            names = sample_names
        elif hasattr(self, 'sample_names_in_'): 
            names = self.sample_names_in_
        else: 
            names = [f"S_{i}" for i in range(n_samples)]
        
        data = {'Sample': names, 't_pred': self.t_pred_.flatten()}
        n_ortho = getattr(self, 'n_ortho', 0)
        
        for i in range(n_ortho):
            data[f't_ortho_{i+1}'] = self.T_ortho_[:, i]
            
        if y_true is not None: 
            data['Class'] = y_true
            
        return pd.DataFrame(data)

    def get_features_df(self, feature_names=None):
        """Exports feature metrics (VIP, Covariance, Correlation).

        Args:
            feature_names (list, optional): Feature names. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame containing feature metrics, 
                sorted by VIP in descending order.
        """
        n_features = len(self.vip_ropls_)
        if feature_names is not None: 
            names = feature_names
        elif hasattr(self, 'feature_names_in_'): 
            names = self.feature_names_in_
        else: 
            names = [f"F_{i}" for i in range(n_features)]
        
        df = pd.DataFrame({
            'Feature': names, 
            'VIP': self.vip_ropls_, 
            'Covariance': self.covariances_,
            'Correlation': self.correlations_, 
            'p_weight': self.p_pred_.flatten()
        })
        return df.sort_values(by='VIP', ascending=False).reset_index(drop=True)