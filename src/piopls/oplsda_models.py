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
from .utils import get_custom_progress


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
        vip_method (str): 'vip4' (ropls default) or 'traditional' (W* based).
    """

    def __init__(
        self, 
        n_ortho=None, 
        cv_folds=7, 
        max_ortho=10, 
        n_perms=100, 
        n_jobs=-1,
        vip_method='vip4'
    ):
        """Initializes the OPLSDA model with specific configurations."""
        self.n_ortho = n_ortho       
        self.cv_folds = cv_folds     
        self.max_ortho = max_ortho
        self.n_perms = n_perms
        self.n_jobs = n_jobs
        self.vip_method = vip_method
        self.label_encoder = LabelEncoder()
        self._is_categorical = False 

    def fit_pipeline(
        self, X, y, run_permutations = True
    ) -> dict:
        """Executes the complete OPLS-DA pipeline in a single convenient call.
        
        Automatically performs model fitting, cross-validation (Q2), and 
        optionally the permutation testing, mimicking the 'ropls' default 
        behavior.
        """
        # Step 1: Base model fitting
        self.fit(X, y)
        
        # Step 2: Cross-validation for Q2
        self.compute_q2(X, y)
        
        # Step 3: Optional Permutation Test
        perm_results: dict = {}
        if run_permutations:
            perm_results = self.permutation_test(X, y)
            
        return perm_results

    def _venetian_blinds_split(self, y_numeric):
        """Splits data using the Stratified Venetian blinds CV strategy."""
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
        """Automatically finds optimal number of orthogonal components."""
        base_model = OPLSDA(
            n_ortho=0, cv_folds=self.cv_folds, vip_method=self.vip_method
        )
        base_model.fit(X, y_numeric)
        base_model.compute_q2(X, y_numeric)
        best_n = 0
        q2_prev = base_model.Q2_
        
        for n in range(1, self.max_ortho + 1):
            temp_model = OPLSDA(
                n_ortho=n, cv_folds=self.cv_folds, vip_method=self.vip_method
            )
            temp_model.fit(X, y_numeric)
            temp_model.compute_q2(X, y_numeric)
            q2_curr = temp_model.Q2_
            
            if q2_curr - q2_prev >= 0.01:
                best_n = n
                q2_prev = q2_curr
            else:
                break
        return max(1, best_n)

    def fit(self, X, y):
        """Fits the OPLS-DA model natively tolerating NaN values."""
        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns.tolist()
        if hasattr(X, 'index'):
            self.sample_names_in_ = X.index.tolist()
            
        X = np.asarray(X, dtype=float)
        y_array = np.asarray(y)
        
        if y_array.dtype.kind in {'U', 'S', 'O'} or len(np.unique(y_array))<=2: 
            y_numeric = self.label_encoder.fit_transform(y_array).astype(float)
            self._is_categorical = True
        else:
            y_numeric = y_array.astype(float)
            self._is_categorical = False
            self.label_encoder.fit(y_numeric) 
            
        n_samples, n_features = X.shape
        if self.n_ortho is None:
            self.n_ortho = self._find_best_n_ortho(X, y_numeric)

        # NaN safe scaling
        self.x_mean_ = np.nanmean(X, axis=0)
        self.x_std_ = np.nanstd(X, axis=0, ddof=1)
        self.x_std_[self.x_std_ == 0] = 1.0 
        E = (X - self.x_mean_) / self.x_std_
        E_orig = E.copy() 

        self.y_mean_ = np.nanmean(y_numeric)
        self.y_std_ = np.nanstd(y_numeric, ddof=1)
        if self.y_std_ == 0: 
            self.y_std_ = 1.0
        f = (y_numeric - self.y_mean_) / self.y_std_

        SS_X_total = np.nansum(E_orig ** 2)
        SS_Y_total = np.nansum((y_numeric - self.y_mean_) ** 2)

        _n_ortho = max(0, self.n_ortho)
        
        self.T_ortho_ = np.zeros((n_samples, _n_ortho)) 
        self.P_ortho_ = np.zeros((n_features, _n_ortho)) 
        self.W_ortho_ = np.zeros((n_features, _n_ortho))
        
        self.R2Y_abs_ = []
        r2x_ortho_list = []

        # ---- Step 0: Initial Predictive Component ----
        w_pred_0 = np.nansum(E * f[:, np.newaxis], axis=0)
        w_pred_0 /= np.linalg.norm(w_pred_0)
        
        valid_mask = ~np.isnan(E)
        t_pred_0 = np.nansum(E * w_pred_0, axis=1) 
        t_pred_0 /= np.sum(valid_mask * (w_pred_0 ** 2), axis=1)
        
        p_pred_0 = np.nansum(E * t_pred_0[:, np.newaxis], axis=0) 
        p_pred_0 /= np.nansum(t_pred_0 ** 2)
        
        c_pred_0 = np.nansum(f * t_pred_0) / np.nansum(t_pred_0 ** 2)
        
        y_pred_0 = (t_pred_0 * c_pred_0) * self.y_std_ + self.y_mean_
        R2Y_cum_0 = 1.0 - np.nansum((y_numeric - y_pred_0)**2) / SS_Y_total
        
        self.R2Y_abs_.append(R2Y_cum_0)
        
        if _n_ortho == 0:
            self.w_pred_ = w_pred_0
            self.t_pred_ = t_pred_0
            self.p_pred_ = p_pred_0
            self.c_pred_ = c_pred_0
            self.R2X_ = (
                np.nansum(t_pred_0**2) * np.nansum(p_pred_0**2)) / SS_X_total
            self.R2Y_ = R2Y_cum_0
            self.RMSEE_ = np.sqrt(np.nanmean((y_numeric - y_pred_0) ** 2))
            self.fitted_values_ = y_pred_0

        # ---- Step 1 to n: Extract Orthogonal Components ----
        for i in range(_n_ortho):
            w = np.nansum(E * f[:, np.newaxis], axis=0)
            w /= np.linalg.norm(w)
            
            valid_mask = ~np.isnan(E)
            t = np.nansum(E * w, axis=1) 
            t /= np.sum(valid_mask * (w ** 2), axis=1)
            
            p = np.nansum(E * t[:, np.newaxis], axis=0) / np.nansum(t ** 2)
            
            w_ortho = p - np.dot(w.T, p) * w
            w_ortho /= np.linalg.norm(w_ortho)
            
            t_ortho = np.nansum(E * w_ortho, axis=1) 
            t_ortho /= np.sum(valid_mask * (w_ortho ** 2), axis=1)
            
            for j in range(i):
                t_prev = self.T_ortho_[:, j]
                proj = np.nansum(t_ortho * t_prev) / np.nansum(t_prev ** 2)
                t_ortho -= proj * t_prev
                
            p_ortho = np.nansum(E * t_ortho[:, np.newaxis], axis=0) 
            p_ortho /= np.nansum(t_ortho ** 2)            
            
            self.T_ortho_[:, i] = t_ortho
            self.P_ortho_[:, i] = p_ortho
            self.W_ortho_[:, i] = w_ortho
            
            # Record explicit R2X of this orthogonal component 
            # (matches ropls perfectly)
            ss_ortho_i = np.nansum(t_ortho**2) * np.nansum(p_ortho**2)
            r2x_ortho_list.append(ss_ortho_i / SS_X_total)

            E = E - np.outer(t_ortho, p_ortho)

            w_pred_k = np.nansum(E * f[:, np.newaxis], axis=0)
            w_pred_k /= np.linalg.norm(w_pred_k)
            
            valid_mask = ~np.isnan(E)
            t_pred_k = np.nansum(E * w_pred_k, axis=1) 
            t_pred_k /= np.sum(valid_mask * (w_pred_k ** 2), axis=1)
            
            for j in range(i + 1):
                t_prev = self.T_ortho_[:, j]
                proj = np.nansum(t_pred_k * t_prev) / np.nansum(t_prev ** 2)
                t_pred_k -= proj * t_prev

            p_pred_k = np.nansum(E * t_pred_k[:, np.newaxis], axis=0) 
            p_pred_k /= np.nansum(t_pred_k ** 2)
            
            c_pred_k = np.nansum(f * t_pred_k) / np.nansum(t_pred_k ** 2)
            
            y_pred_k = (t_pred_k * c_pred_k) * self.y_std_ + self.y_mean_
            R2Y_cum_k = 1.0 - np.nansum((y_numeric - y_pred_k)**2) / SS_Y_total
            
            self.R2Y_abs_.append(R2Y_cum_k)
            
            if i == _n_ortho - 1:
                self.w_pred_ = w_pred_k
                self.t_pred_ = t_pred_k
                self.p_pred_ = p_pred_k
                self.c_pred_ = c_pred_k
                self.RMSEE_ = np.sqrt(np.nanmean((y_numeric - y_pred_k) ** 2))
                self.fitted_values_ = y_pred_k

        # ---- Process Metrics matching ropls tables ----
        if _n_ortho > 0:
            # 1. R2X
            r2x_pred = ( 
                np.nansum(self.t_pred_**2) * np.nansum(
                    self.p_pred_**2)) / SS_X_total
            self.R2X_comp_ = [r2x_pred] + r2x_ortho_list
            self.R2X_cum_list_ = list(np.cumsum(self.R2X_comp_))

            # 2. R2Y (ropls specific incremental tracking for ortho components)
            self.R2Y_comp_ = []
            r2y_ropls_cum = []
            for i in range(len(self.R2Y_abs_)):
                if i == 0:
                    self.R2Y_comp_.append(self.R2Y_abs_[0])
                    r2y_ropls_cum.append(self.R2Y_abs_[0])
                else:
                    self.R2Y_comp_.append(self.R2Y_abs_[i] - self.R2Y_abs_[i-1])
                    r2y_ropls_cum.append(self.R2Y_abs_[i] - self.R2Y_abs_[0])
            self.R2Y_cum_list_ = r2y_ropls_cum
        else:
            self.R2X_comp_ = [self.R2X_]
            self.R2X_cum_list_ = [self.R2X_]
            self.R2Y_comp_ = [self.R2Y_abs_[-1]]
            self.R2Y_cum_list_ = [self.R2Y_abs_[-1]]

        # ---- Process Categorical Fitted Labels ----
        if self._is_categorical and hasattr(self, 'fitted_values_'):
            n_classes = len(self.label_encoder.classes_)
            y_pred_idx = np.clip(
                np.round(self.fitted_values_).astype(int), 0, n_classes - 1
            )
            inv_trans = self.label_encoder.inverse_transform
            self.fitted_class_ = inv_trans(y_pred_idx)

        # ---- Calculate VIP metrics based on user configuration ----
        if self.vip_method == 'vip4':
            sxp = np.nansum(self.t_pred_**2) * np.nansum(self.p_pred_**2)
            if _n_ortho > 0:
                sxo = sum(
                    np.nansum(self.T_ortho_[:, j]**2) * np.nansum(self.P_ortho_[:, j]**2) 
                    for j in range(_n_ortho)
                )
            else:
                sxo = 0
                
            ssx_cum = sxp + sxo
            syp = np.nansum(self.t_pred_**2) * (self.c_pred_**2)
            ssy_cum = syp

            kp = n_features / ((sxp / ssx_cum) + (syp / ssy_cum))
            p_norm = self.p_pred_ / np.sqrt(np.nansum(self.p_pred_**2))
            
            term_x = (p_norm ** 2 * sxp) / ssx_cum
            term_y = (p_norm ** 2 * syp) / ssy_cum
            self.vip_ropls_ = np.sqrt(kp * (term_x + term_y))
        else:
            if _n_ortho > 0:
                W_all = np.column_stack((self.w_pred_, self.W_ortho_))
                P_all = np.column_stack((self.p_pred_, self.P_ortho_))
                W_star = np.dot(W_all, np.linalg.inv(np.dot(P_all.T, W_all)))
                w_star_pred = W_star[:, 0]
            else:
                w_star_pred = self.w_pred_
                
            w_star_pred /= np.linalg.norm(w_star_pred)
            self.vip_ropls_ = np.sqrt(n_features * (w_star_pred ** 2))

        # ---- Calculate precise Covariance/Correlation for S-Plot ----
        covariances = np.zeros(n_features)
        correlations = np.zeros(n_features)
        
        X_centered = X - self.x_mean_
        
        for j in range(n_features):
            mask = ~np.isnan(X_centered[:, j])
            t_valid = self.t_pred_[mask]
            x_valid = X_centered[mask, j]
            
            if len(t_valid) > 1:
                cov_mat = np.cov(t_valid, x_valid)
                covariances[j] = cov_mat[0, 1]
                
                std_t = np.std(t_valid, ddof=1)
                std_x = np.std(x_valid, ddof=1)
                if std_t > 0 and std_x > 0:
                    correlations[j] = covariances[j] / (std_t * std_x)
                    
        self.covariances_ = covariances
        self.correlations_ = correlations

        # ---- Calculate Outlier Diagnostics (SD and OD) ----
        # Orthogonal Distance (OD): Norm of the final residual matrix E
        E_res = E_orig.copy()
        pred_hat = np.outer(self.t_pred_, self.p_pred_)
        E_res = E_res - pred_hat
        for i in range(_n_ortho):
            ortho_hat = np.outer(self.T_ortho_[:, i], self.P_ortho_[:, i])
            E_res = E_res - ortho_hat
            
        # NaN-safe calculation of OD
        sq_res = E_res ** 2
        self.OD_ = np.sqrt(np.nanmean(sq_res, axis=1) * n_features)

        # Score Distance (SD): Mahalanobis distance in score space
        if _n_ortho > 0:
            T_all = np.column_stack((self.t_pred_, self.T_ortho_))
        else:
            T_all = self.t_pred_[:, np.newaxis]
            
        T_var = np.var(T_all, axis=0, ddof=1)
        T_var[T_var == 0] = 1e-10  # Prevent division by zero
        
        self.SD_ = np.sqrt(np.sum((T_all ** 2) / T_var, axis=1))

        # ---- Calculate Outlier Limits (Moved from Plotting) ----
        from scipy.stats import chi2, norm
        
        # SD 95% Limit (Chi-Square)
        n_comps = 1 + getattr(self, 'n_ortho', 0)
        self.sd_limit_ = np.sqrt(chi2.ppf(0.95, df=n_comps))
        
        # OD 95% Limit (Jackson-Mudholkar approximation)
        od_vals = self.OD_
        if len(od_vals) > 1:
            od_23 = od_vals ** (2 / 3)
            mu_od = np.mean(od_23)
            std_od = np.std(od_23, ddof=1)
            self.od_limit_ = (mu_od + norm.ppf(0.95) * std_od) ** (3 / 2)
        else:
            self.od_limit_ = np.max(od_vals) * 1.1

        return self

    def _predict_continuous(self, X):
        """Internal method to predict continuous target values safely."""
        X = np.asarray(X, dtype=float)
        E = (X - self.x_mean_) / self.x_std_
        
        for i in range(self.n_ortho):
            valid_mask = ~np.isnan(E)
            w_o_sq = self.W_ortho_[:, i] ** 2
            t_ortho = np.nansum(E * self.W_ortho_[:, i], axis=1) 
            t_ortho /= np.sum(valid_mask * w_o_sq, axis=1)
            E -= np.outer(t_ortho, self.P_ortho_[:, i])
            
        valid_mask = ~np.isnan(E)
        w_p_sq = self.w_pred_ ** 2
        t_pred = np.nansum(E * self.w_pred_, axis=1) 
        t_pred /= np.sum(valid_mask * w_p_sq, axis=1)
        
        y_pred_num = (t_pred * self.c_pred_) * self.y_std_ + self.y_mean_
        return y_pred_num

    def predict(self, X):
        """Predicts class labels or continuous values for the input data."""
        y_pred_num = self._predict_continuous(X)
        if self._is_categorical:
            n_classes = len(self.label_encoder.classes_)
            y_pred_idx = np.clip(
                np.round(y_pred_num).astype(int), 0, n_classes - 1
            )
            return self.label_encoder.inverse_transform(y_pred_idx)
        return y_pred_num

    def compute_q2(self, X, y):
        """Calculates the Q2 cross-validated predictive ability."""
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
                model_cv = OPLSDA(
                    n_ortho=n, 
                    cv_folds=self.cv_folds, 
                    vip_method=self.vip_method
                )
                model_cv.fit(X_train, y_train)
                y_cv_preds[n][test_idx] = model_cv._predict_continuous(X_test)
                
        SS_Y = np.nansum((y_num - np.nanmean(y_num)) ** 2)
        PRESS_list = [
            np.nansum((y_num - y_cv_preds[n]) ** 2) 
            for n in range(n_ortho_plus_1)
        ]
        
        self.Q2_abs_ = [1.0 - (press / SS_Y) for press in PRESS_list]
        self.Q2_comp_ = []
        self.Q2_cum_list_ = []
        
        for i in range(len(self.Q2_abs_)):
            if i == 0:
                self.Q2_comp_.append(self.Q2_abs_[0])
                self.Q2_cum_list_.append(self.Q2_abs_[0])
            else:
                self.Q2_comp_.append(self.Q2_abs_[i] - self.Q2_abs_[i-1])
                # Mirrors ropls: Q2(cum) for ortho is relative to step 0
                self.Q2_cum_list_.append(self.Q2_abs_[i] - self.Q2_abs_[0])
                
        self.Q2_ = self.Q2_abs_[-1]
        return self.Q2_

    def _single_permutation(self, X, y_numeric, n_ortho, cv_folds):
        """Runs a single iteration of the permutation test."""
        np.random.seed()
        y_perm = np.random.permutation(y_numeric) 
        model_perm = OPLSDA(
            n_ortho=n_ortho, cv_folds=cv_folds, vip_method=self.vip_method
        )
        model_perm.fit(X, y_perm)
        q2 = model_perm.compute_q2(X, y_perm)
        return model_perm.R2Y_abs_[-1], q2

    def permutation_test(self, X, y, n_perms=None, n_jobs=None):
        """Perform a permutation test to assess model significance."""
        n_perms = n_perms or self.n_perms
        n_jobs = n_jobs or self.n_jobs
        
        X_mat = X.values if hasattr(X, 'values') else np.array(X)
        y_arr = y.values if hasattr(y, 'values') else np.array(y)
        
        if self._is_categorical:
            y_numeric = self.label_encoder.transform(y_arr).astype(float)
        else:
            y_numeric = y_arr.astype(float)

        print(
            f"Starting parallel permutation test ({n_perms} permutations)..."
        )

        current_n_ortho = getattr(self, 'n_ortho', 1) 
        current_cv_folds = getattr(self, 'cv_folds', 7)
        
        results_gen = Parallel(n_jobs=n_jobs, return_as='generator')(
            delayed(self._single_permutation)(
                X_mat, y_numeric, current_n_ortho, current_cv_folds)
            for _ in range(n_perms)
        )

        pbar = get_custom_progress(
            results_gen, total=n_perms, desc="Permutation Test",
            color="#2CA02C", bar_length=60
        )
        results = list(pbar)

        perms_R2Y = [res[0] for res in results]
        perms_Q2 = [res[1] for res in results]

        valid_perms = len(perms_R2Y)
        if valid_perms == 0:
            raise ValueError("All permutation iterations failed to converge.")

        orig_R2Y = getattr(self, 'R2Y_abs_', [0])[-1]
        orig_Q2 = getattr(self, 'Q2_', 0)

        count_R2Y = sum(1 for val in perms_R2Y if val >= orig_R2Y)
        count_Q2 = sum(1 for val in perms_Q2 if val >= orig_Q2)

        p_R2Y = (count_R2Y + 1) / (valid_perms + 1)
        p_Q2 = (count_Q2 + 1) / (valid_perms + 1)

        self.p_R2Y_ = p_R2Y
        self.p_Q2_ = p_Q2
        
        return {
            'orig_R2Y': orig_R2Y,
            'orig_Q2Y': orig_Q2,
            'perm_R2Y': perms_R2Y,
            'perm_Q2Y': perms_Q2,
            'p_R2Y': p_R2Y,
            'p_Q2Y': p_Q2,
            'valid_perms': valid_perms
        }

    # ==========================================
    # DataFrame Export Methods (ropls aligned)
    # ==========================================
    def get_model_info_df(self):
        """Exports the global model metadata (Matches ropls getSummaryDF)."""
        if not hasattr(self, 'Q2_'):
            raise ValueError("Model must compute Q2 first.")
            
        data = {
            'N_Predictive': [1],
            'N_Ortho': [self.n_ortho],
            'R2X(cum)': [self.R2X_cum_list_[-1]],
            'R2Y(cum)': [self.R2Y_abs_[-1]],
            'Q2(cum)': [self.Q2_abs_[-1]],
            'RMSEE': [self.RMSEE_]
        }
        
        if hasattr(self, 'p_R2Y_'):
            data['pR2Y'] = [self.p_R2Y_]
            data['pQ2'] = [self.p_Q2_]
            
        return pd.DataFrame(data)

    def get_summary_df(self):
        """Exports the step-wise incremental and cumulative metrics."""
        if not hasattr(self, 'Q2_comp_'):
            raise ValueError("Model must compute Q2 first.")
            
        n_ortho = getattr(self, 'n_ortho', 0)
        components = ['Predictive (p1)'] + [
            f'Orthogonal (o{i+1})' for i in range(n_ortho)
        ]
        
        return pd.DataFrame({
            'Component': components, 
            'R2X': self.R2X_comp_, 
            'R2Y': self.R2Y_comp_, 
            'Q2': self.Q2_comp_,
            'R2X(cum)': self.R2X_cum_list_,
            'R2Y(cum)': self.R2Y_cum_list_,
            'Q2(cum)': self.Q2_cum_list_
        })

    def get_scores_df(self, sample_names=None, y_true=None):
        """Exports sample scores, fitted values/classes, and evaluation metrics."""
        n_samples = self.t_pred_.shape[0]
        if sample_names is not None: 
            names = sample_names
        elif hasattr(self, 'sample_names_in_'): 
            names = self.sample_names_in_
        else: 
            names = [f"S_{i}" for i in range(n_samples)]
        
        data = {'Sample': names, 't_pred (p1)': self.t_pred_.flatten()}
        n_ortho = getattr(self, 'n_ortho', 0)
        
        for i in range(n_ortho):
            data[f't_ortho (o{i+1})'] = self.T_ortho_[:, i]
            
        if y_true is not None: 
            data['True_Class'] = np.asarray(y_true)
            
        if hasattr(self, 'fitted_values_'):
            data['Fitted_Value'] = self.fitted_values_.flatten()
        if hasattr(self, 'fitted_class_'):
            data['Fitted_Class'] = self.fitted_class_.flatten()
            
        df = pd.DataFrame(data)
        
        if 'True_Class' in df.columns and 'Fitted_Class' in df.columns:
            df['Match_Status'] = np.where(
                df['True_Class'] == df['Fitted_Class'], 
                'Matched', 
                'Mismatched'
            )
            
        return df

    def get_features_df(self, feature_names=None):
        """Exports feature selection metrics (VIP, Covariance, Correlation)."""
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
            'Covariance (p1)': self.covariances_,
            'Correlation (pcorr1)': self.correlations_, 
            'Loading_Weight': self.p_pred_.flatten()
        })
        return df.sort_values(by='VIP', ascending=False).reset_index(drop=True)

    def get_outlier_df(self, sample_names=None, y_true=None):
        """Exports Score/Orthogonal Distances and outlier limit flags."""
        n_samples = getattr(self, 'SD_', []).shape[0] if hasattr(
            self, 'SD_') else 0
            
        if sample_names is not None: 
            names = sample_names
        elif hasattr(self, 'sample_names_in_'): 
            names = self.sample_names_in_
        else: 
            names = [f"S_{i}" for i in range(n_samples)]
            
        df = pd.DataFrame({
            'Sample': names,
            'Score_Distance': getattr(self, 'SD_', []),
            'Orthogonal_Distance': getattr(self, 'OD_', [])
        })
        
        # Add flags for limit exceedance
        if hasattr(self, 'sd_limit_') and hasattr(self, 'od_limit_'):
            df['Exceeds_SD_Limit'] = df['Score_Distance'] > self.sd_limit_
            df['Exceeds_OD_Limit'] = df['Orthogonal_Distance'] > self.od_limit_
        
        if y_true is not None: 
            df['True_Class'] = np.asarray(y_true)
            
        return df