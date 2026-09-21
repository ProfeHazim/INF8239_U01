import pandas as pd
from sklearn.datasets import load_breast_cancer

def load_data():
    """Carga el dataset público de cáncer de mama para el análisis SVM."""
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df, data.target_names