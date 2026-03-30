# src/piopls/oplsda_plotting.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse, Patch
from matplotlib.transforms import Affine2D

class OPLSDA_Visualizer:
    """
    Visualization suite generating the 5 standard diagnostic plots for OPLS-DA:
    (A) Model Overview, (B) X-Score Plot, (C) Permutation Test, (D) S-Plot, and (E) VIP Bar Plot.
    Consumes pandas DataFrames exported by the OPLSDA model.
    """
    def __init__(
        self, model, X=None, y=None, feature_names=None, sample_names=None, 
        vip_threshold=1.0, top_n_vip=20, custom_palette=None):
        
        self.model = model
        self.feature_names = feature_names
        self.sample_names = sample_names
        
        if hasattr(model, 'label_encoder') and len(model.label_encoder.classes_) > 0:
            self.class_names = {i: name for i, name in enumerate(model.label_encoder.classes_)}
            self.y_groups = y if y is not None else [self.class_names[0]] * len(model.t_pred_) 
        else:
            self.class_names = {0: 'Class 0', 1: 'Class 1'}
            self.y_groups = ['Class 1' if val > np.median(y) else 'Class 0' for val in y] if y is not None else ['Class 0'] * len(model.t_pred_)

        self.vip_threshold = vip_threshold
        self.top_n_vip = top_n_vip
        
        if custom_palette is not None:
            self.palette = custom_palette
        else:
            keys = list(set(self.y_groups))
            self.palette = {keys[0]: 'tab:blue', keys[1]: 'tab:red'} if len(keys) >= 2 else {keys[0]: 'tab:blue'}

    def _draw_confidence_ellipse(self, x, y, ax, n_std=2.0, **kwargs):
        if x.size != y.size or x.size < 3: return
        cov = np.cov(x, y)
        if cov[0,0] == 0 or cov[1,1] == 0: return
        pearson = np.clip(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]), -1.0, 1.0) 
        ellipse = Ellipse((0, 0), width=np.sqrt(1 + pearson) * 2, height=np.sqrt(1 - pearson) * 2, **kwargs)
        scale_x, scale_y = np.sqrt(cov[0, 0]) * n_std, np.sqrt(cov[1, 1]) * n_std
        transf = Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(np.mean(x), np.mean(y))
        ellipse.set_transform(transf + ax.transData)
        ax.add_patch(ellipse)

    def _format_ax(self, ax, xlabel, ylabel, title):
        ax.set_xlabel(xlabel, fontsize=11, fontweight='normal')
        ax.set_ylabel(ylabel, fontsize=11, fontweight='normal')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
        ax.tick_params(axis='both', labelsize=11)
        sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)

    def _add_legend(self, ax, title="", loc="best", ncol=1):
        ax.legend(
            title=title, ncol=ncol, frameon=True, shadow=True, fontsize=10,
            borderpad=0.4, facecolor="white", loc=loc)

    def plot_all(self, perm_results=None, save_path=None):
        fig = plt.figure(figsize=(15, 12))
        gs = fig.add_gridspec(2, 3)
        ax_ov, ax_sc, ax_vp = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0:, 2])
        ax_pm, ax_sp = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

        self.plot_model_overview(ax=ax_ov)
        self.plot_score(ax=ax_sc)
        self.plot_splot(ax=ax_sp)
        self.plot_vip_bar(ax=ax_vp)
        
        if perm_results: self.plot_permutations(perm_results, ax=ax_pm)
        else: ax_pm.axis('off')

        fig.tight_layout(h_pad=4.0, w_pad=3.0)
        if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_model_overview(self, ax=None):
        df_summary = self.model.get_summary_df()
        labels = df_summary['Component'].tolist()
        r2y_vals = df_summary['R2Y'].tolist()
        q2_vals  = df_summary['Q2'].tolist()
        
        if ax is None: _, ax = plt.subplots(figsize=(max(4.5, len(labels) * 1.5), 3))
            
        x = np.arange(len(labels))
        width = 0.35

        ax.bar(x - width/2, r2y_vals, width, label='R2Y', color='#6BAED6', edgecolor='k', linewidth=0.5)
        ax.bar(x + width/2, q2_vals,  width, label='Q2',  color='#2171B5', edgecolor='k', linewidth=0.5)

        def add_labels(vals, offset):
            for i, v in enumerate(vals):
                if v == 0: continue
                y_pos = v + 0.02 if v >= 0 else v - 0.05
                ax.text(i + offset, y_pos, f'{v:.3f}', ha='center', va='bottom' if v >= 0 else 'top', fontsize=9)

        add_labels(r2y_vals, -width/2)
        add_labels(q2_vals, width/2)

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        
        max_val = max([max(r2y_vals), max(q2_vals)])
        min_val = min([min(q2_vals), 0])
        ax.set_ylim(min_val - 0.1 if min_val < 0 else 0, min(1.2, max_val * 1.35))
        if min_val < 0: ax.axhline(0, color='k', linewidth=0.8)
        
        self._format_ax(ax, "", "", "Model Overview")
        self._add_legend(ax, ncol=2, loc='upper left')

    def plot_score(self, ax=None):
        if ax is None: _, ax = plt.subplots(figsize=(4.5, 3))
        
        df_scores = self.model.get_scores_df(sample_names=self.sample_names, y_true=self.y_groups)
        df_summary = self.model.get_summary_df()
        
        t1 = df_scores['t_pred']
        p_explan = df_summary.loc[df_summary['Component'] == 'p1', 'R2X'].values[0] * 100
        
        if 't_ortho_1' in df_scores.columns:
            o1 = df_scores['t_ortho_1']
            o_explan = df_summary.loc[df_summary['Component'] == 'o1', 'R2X'].values[0] * 100
        else:
            o1 = np.zeros_like(t1)
            o_explan = 0.0
            
        df_scores['t_ortho_1_plot'] = o1
            
        sns.scatterplot(data=df_scores, x='t_pred', y='t_ortho_1_plot', hue='Class', 
                        palette=self.palette, edgecolor='k', linewidth=0.5, s=40, ax=ax)
        
        for cls_name in np.unique(df_scores['Class']):
            sub = df_scores[df_scores['Class'] == cls_name]
            if cls_name in self.palette:
                color = self.palette[cls_name]
                self._draw_confidence_ellipse(
                    sub['t_pred'], sub['t_ortho_1_plot'], ax, n_std=2.0, 
                    facecolor=color, edgecolor=color, alpha=0.3, linewidth=1)

        self._format_ax(ax, f"T Score [1]\n({p_explan:.1f} %)", f"Orthogonal T Score [1]\n({o_explan:.1f} %)", "X-Score Plot")
        self._add_legend(ax, ncol=2, loc='best')

    def plot_permutations(self, perm_results, ax=None):
        if ax is None: _, ax = plt.subplots(figsize=(4.5, 3))
        perms_r2y, perms_q2 = perm_results.get('perms_R2Y', []), perm_results.get('perms_Q2', [])
        real_r2y, real_q2 = perm_results.get('R2Y_real', 0), perm_results.get('Q2_real', 0)
        
        if len(perms_r2y) > 0:
            bins = np.histogram_bin_edges(np.concatenate([perms_r2y, perms_q2]), bins=30)
            ax.hist(perms_r2y, bins=bins, color='tab:red', edgecolor='k', linewidth=0.5, label='Perm R2Y')
            ax.hist(perms_q2,  bins=bins, color='tab:blue', edgecolor='k', linewidth=0.5, label='Perm Q2')
            
        maxY = ax.get_ylim()[1]
        xmin, xmax = ax.get_xlim()
        needed_xmax = max(real_r2y, real_q2, xmax)
        ax.set_xlim(xmin, needed_xmax + (needed_xmax - xmin) * 0.15)
        
        ax.axvline(real_r2y, color='tab:red', linestyle='--', linewidth=1.0, zorder=0)
        ax.axvline(real_q2, color='tab:blue', linestyle='--', linewidth=1.0, zorder=0)

        arrowprops = dict(arrowstyle="-|>", connectionstyle="arc3, rad=-0.15", color="k", linewidth=1.0)
        bbox_props = dict(boxstyle="round", facecolor="white", edgecolor="k", pad=0.2)
        
        ax.annotate(f"R2Y: {real_r2y:.3f}\nP = {perm_results.get('p_R2Y', 0):.3f}", 
                    xy=(real_r2y, 0), xytext=(real_r2y, maxY * 0.75),
                    ha="center", va="center", fontsize=8, bbox=bbox_props, arrowprops=arrowprops)
        ax.annotate(f"Q2: {real_q2:.3f}\nP = {perm_results.get('p_Q2', 0):.3f}", 
                    xy=(real_q2, 0), xytext=(real_q2, maxY * 0.5),
                    ha="center", va="center", fontsize=8, bbox=bbox_props, arrowprops=arrowprops)

        self._format_ax(ax, "Permutations", "Frequency", "Permutation Test")
        self._add_legend(ax, ncol=1, loc='best')

    def plot_splot(self, ax=None):
        if ax is None: _, ax = plt.subplots(figsize=(4.5, 3))
        
        df_features = self.model.get_features_df(feature_names=self.feature_names)
        df_features['Cat'] = np.where(df_features['VIP'] >= self.vip_threshold, f'VIP>={self.vip_threshold}', f'VIP<{self.vip_threshold}')
        palette = {f'VIP>={self.vip_threshold}': 'tab:red', f'VIP<{self.vip_threshold}': 'tab:blue'}
        
        sns.scatterplot(data=df_features, x='Covariance', y='Correlation', hue='Cat', 
                        palette=palette, edgecolor='k', linewidth=0.5, s=35, ax=ax)
        
        ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
        ax.axvline(0, color='k', linestyle='--', linewidth=0.5)
        self._format_ax(ax, "Covariance", "Pearson's r", "S-Plot")
        self._add_legend(ax, ncol=1, loc='best')

    def plot_vip_bar(self, ax=None):
        df_features = self.model.get_features_df(feature_names=self.feature_names)
        top_n = min(self.top_n_vip, len(df_features))
        if ax is None: _, ax = plt.subplots(figsize=(4.5, top_n * 0.22))
        
        df_top = df_features.head(top_n).sort_values(by='VIP', ascending=True)
        df_top['Color'] = df_top['Correlation'].apply(lambda c: 'tab:red' if c > 0 else 'tab:blue')
        
        bars = ax.barh(df_top['Feature'], df_top['VIP'], color=df_top['Color'], edgecolor='k', linewidth=0.5, height=0.8)
        
        label_jitter = (df_top['VIP'].max() - df_top['VIP'].min()) / 50 if len(df_top) > 1 else 0.1
        for bar in bars:
            ax.text(bar.get_width() + label_jitter, bar.get_y() + bar.get_height()/2, 
                    f'{bar.get_width():.2f}', va='center', ha='left', fontsize=10)
                    
        self._format_ax(ax, "VIP", "", "VIP Bar Plot")
        ax.spines["left"].set_position(("data", 0))
        ax.tick_params(left=False)
        
        legend_elements = [
            Patch(facecolor='tab:red', edgecolor='k', label='Positive'), 
            Patch(facecolor='tab:blue', edgecolor='k', label='Negative')]
        ax.legend(
            handles=legend_elements, title="Correlation", 
            ncol=1, frameon=True, shadow=True, fontsize=10, borderpad=0.4, 
            facecolor="white", loc="lower right")