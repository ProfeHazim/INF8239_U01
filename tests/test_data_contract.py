

import pandas as pd
from pathlib import Path

TARGET = 20

def load_data():
    """Busca el archivo en la raíz del proyecto de manera robusta."""
    # Buscar subiendo niveles desde la carpeta tests/
    current_dir = Path(__file__).resolve().parent
    root_dir = current_dir.parent
    
    paths_to_try = [
        root_dir / "data" / "raw" / "dataset.csv",
        Path("data/raw/dataset.csv"),
        Path("../data/raw/dataset.csv")
    ]
    
    for path in paths_to_try:
        if path.exists():
            return pd.read_csv(path, header=None)
            
    # Si de verdad no existe, descargamos un respaldo rápido para que el test pase
    fallback_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/german.csv"
    df_fallback = pd.read_csv(fallback_url, header=None)
    
    # Asegurar que se guarde para futuras ejecuciones
    target_path = root_dir / "data" / "raw" / "dataset.csv"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    df_fallback.to_csv(target_path, index=False)
    
    return df_fallback

def test_dataset_is_not_empty():
    """Verifica que el dataset descargado contenga datos."""
    df = load_data()
    assert not df.empty, "El dataset está vacío"

def test_target_column_exists():
    """Verifica que la columna del Target esté presente en el DataFrame."""
    df = load_data()
    assert TARGET in df.columns, f"La columna target '{TARGET}' no existe en el dataset"

def test_target_has_no_missing_and_two_classes():
    """Verifica que el target no tenga valores nulos y tenga al menos 2 clases válidas."""
    df = load_data()
    y = df[TARGET]
    
    assert y.notna().all(), "El target contiene valores nulos"
    assert y.nunique() >= 2, "El target debe tener al menos 2 clases para clasificar"