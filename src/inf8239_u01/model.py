from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

def train_svm_model(df):
    """Entrena un modelo SVM con los datos proporcionados."""
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Dividir en entrenamiento (80%) y prueba (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Inicializar y entrenar el modelo SVM
    model = SVC(kernel='linear', random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluar
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    return model, acc, classification_report(y_test, y_pred)