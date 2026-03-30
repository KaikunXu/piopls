# tests/test_oplsda.py
import os
import pytest
import numpy as np
import pandas as pd
from piopls import OPLSDA, OPLSDA_Visualizer

@pytest.fixture
def dummy_data():
    """
    Generate virtual metabolomics data for testing.
    Contains 20 samples (two groups) and 15 features, with standard Pandas row/col names.
    """
    np.random.seed(42)
    n_samples = 20
    n_features = 15
    
    # Randomly generate base data
    X = np.random.rand(n_samples, n_features)
    y = np.array([0] * 10 + [1] * 10)
    
    # Artificially create differential features (Biomarkers)
    X[:10, 0] += 2.0
    X[10:, 1] += 2.0
    
    # Wrap as Pandas DataFrame and Series
    feature_names = [f"Metabolite_{i}" for i in range(n_features)]
    sample_names = [f"Sample_{i}" for i in range(n_samples)]
    
    df_X = pd.DataFrame(X, columns=feature_names, index=sample_names)
    df_y = pd.Series(y, index=sample_names, name="Group")
    
    return df_X, df_y

def test_model_fit_and_basic_metrics(dummy_data):
    """Test basic model fitting and R2X/R2Y metric calculation."""
    df_X, df_y = dummy_data
    model = OPLSDA(n_ortho=1, cv_folds=3)
    model.fit(df_X, df_y)
    
    assert hasattr(model, 'R2Y_'), "Model should contain R2Y_ attribute after fitting."
    assert hasattr(model, 'R2X_'), "Model should contain R2X_ attribute after fitting."
    assert model.R2Y_ > 0, "R2Y should be greater than 0."

def test_compute_q2_and_summary_df(dummy_data):
    """Test cross-validation Q2 calculation and summary dataframe output."""
    df_X, df_y = dummy_data
    model = OPLSDA(n_ortho=1, cv_folds=3)
    model.fit(df_X, df_y)
    
    # Calling summary_df before computing Q2 should raise an error
    with pytest.raises(ValueError, match="Model must compute Q2 first"):
        model.get_summary_df()
        
    q2 = model.compute_q2(df_X, df_y)
    assert isinstance(q2, float), "Q2 should return a float value."
    
    df_summary = model.get_summary_df()
    assert isinstance(df_summary, pd.DataFrame)
    assert 'Q2' in df_summary.columns
    assert len(df_summary) == 2, "With n_ortho=1, there should be 2 rows (p1 and o1)."

def test_features_df_and_vip_ranking(dummy_data):
    """Test feature output table, verifying VIP ranking and variable name extraction."""
    df_X, df_y = dummy_data
    model = OPLSDA(n_ortho=1, cv_folds=3)
    model.fit(df_X, df_y)
    
    df_features = model.get_features_df()
    
    assert 'VIP' in df_features.columns
    assert 'Feature' in df_features.columns
    assert 'Covariance' in df_features.columns
    
    assert df_features['Feature'].iloc[0].startswith("Metabolite_"), "Failed to extract original feature names."
    
    vips = df_features['VIP'].values
    assert all(vips[i] >= vips[i+1] for i in range(len(vips)-1)), "Features DataFrame is not sorted by VIP descending."
    
    top_feature = df_features['Feature'].iloc[0]
    assert top_feature in ["Metabolite_0", "Metabolite_1"], "The most differential feature is not at the top."
    assert df_features['VIP'].iloc[0] > 1.0, "The VIP value of the Top 1 feature should be greater than 1.0."

def test_scores_df_metadata(dummy_data):
    """Test sample score table, verifying sample name memorization function."""
    df_X, df_y = dummy_data
    model = OPLSDA(n_ortho=1, cv_folds=3)
    model.fit(df_X, df_y)
    
    df_scores = model.get_scores_df(y_true=df_y.values)
    
    assert 'Sample' in df_scores.columns
    assert 't_pred' in df_scores.columns
    assert 't_ortho_1' in df_scores.columns
    
    assert df_scores['Sample'].iloc[0] == "Sample_0", "Failed to extract original sample names."

def test_permutation_test(dummy_data):
    """Test multi-process permutation test function."""
    df_X, df_y = dummy_data
    model = OPLSDA(n_ortho=1, cv_folds=3)
    model.fit(df_X, df_y) 
    
    perm_res = model.permutation_test(df_X, df_y, n_perms=5, n_jobs=1)
    
    assert 'p_R2Y' in perm_res
    assert 'p_Q2' in perm_res
    assert len(perm_res['perms_R2Y']) == 5, "The length of permutation history records does not match."

def test_sacurine_example_data(tmp_path):
    """
    Test the entire workflow using the built-in Sacurine dataset.
    Uses tmp_path to ensure generated plots do not pollute the workspace.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, "src", "piopls", "data")
    
    path_X = os.path.join(data_dir, "sacurine_X.csv")
    path_Y = os.path.join(data_dir, "sacurine_Y.csv")
    
    if not os.path.exists(path_X) or not os.path.exists(path_Y):
        pytest.skip(f"Example data not found at {data_dir}. Skipping sacurine test.")
        
    df_X = pd.read_csv(path_X, index_col=0)
    df_Y = pd.read_csv(path_Y, index_col=0)
    y_labels = df_Y.iloc[:, 0].values
    
    assert df_X.shape[0] > 0 and df_X.shape[1] > 0, "Loaded X matrix is empty."
    
    model = OPLSDA(cv_folds=7)
    model.fit(df_X, y_labels)
    model.compute_q2(df_X, y_labels)
    
    df_summary = model.get_summary_df()
    assert not df_summary.empty, "Summary DataFrame is empty."
    
    perm_results = model.permutation_test(df_X, y_labels, n_perms=10, n_jobs=-1)
    assert 'p_R2Y' in perm_results and 'p_Q2' in perm_results
    
    vis = OPLSDA_Visualizer(
        model=model,
        y=y_labels, 
        vip_threshold=1.0,
        top_n_vip=15
    )
    
    save_path = os.path.join(tmp_path, "test_oplsda_diagnostic_plots.png")
    vis.plot_all(perm_results=perm_results, save_path=save_path)
    
    assert os.path.exists(save_path), "Failed to generate and save the diagnostic plots."
    
    df_features = model.get_features_df()
    assert len(df_features) == df_X.shape[1], "Features DataFrame length mismatch."