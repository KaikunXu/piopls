# src/piopls/datasets.py
import os
import pandas as pd

def load_sacurine():
    """
    Load the benchmark 'sacurine' dataset (Human Urine Metabolomics).
    
    Returns:
        X (np.ndarray): Feature matrix (183 samples x 109 metabolites).
        y (np.ndarray): Target categorical vector (Gender).
        feature_names (list): List of metabolite names.
        sample_names (list): List of unique sample IDs.
    """
    module_path = os.path.dirname(__file__)
    x_path = os.path.join(module_path, 'data', 'sacurine_X.csv')
    y_path = os.path.join(module_path, 'data', 'sacurine_Y.csv')
    
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(
            "Sacurine dataset files not found. Ensure 'data/' folder contains .csv files."
        )
    
    # Use index_col=0 to set 'sample_id' column as DataFrame index
    df_X = pd.read_csv(x_path, index_col=0)
    df_y = pd.read_csv(y_path, index_col=0)
    
    # Assume that the sample IDs in X and Y are the same and in the same order
    if not (df_X.index == df_y.index).all():
        raise ValueError("Critical Error: Sample IDs in X and Y do not match or are out of order!")
    
    # Extract feature matrix X (remove sample ID column)
    X = df_X.values
    # Extract target vector (Gender) from the first column of Y
    y = df_y.iloc[:, 0].values 
    
    feature_names = df_X.columns.tolist()
    sample_names = df_X.index.tolist()
    
    return X, y, feature_names, sample_names