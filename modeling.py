# modeling.py
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from preprocessing import data_preprocessing, feature_scaling

def train_models(data_path):
    """
    Fungsi utama untuk melatih model Logistic Regression (klasifikasi) dan K-Means (clustering).
    Args:
        data_path (str): Path ke file dataset CSV.
    Returns:
        Tuple: Model Logistic Regression, K-Means, dan scaler.
    """
    
    # 1. PREPROCESSING DATA
    # Memuat dan memproses data menggunakan fungsi dari preprocessing.py
    df = data_preprocessing(data_path)
    
    # Memisahkan fitur (X) dan target (y)
    X = df.drop('Churn', axis=1)  # Hapus kolom target untuk fitur
    y = df['Churn']               # Ambil kolom target saja
    
    # 2. FEATURE SCALING
    # Melakukan scaling pada fitur numerik (standarisasi)
    X_scaled, scaler = feature_scaling(X)
    
    # 3. TRAIN-TEST SPLIT
    # Membagi data menjadi training (80%) dan testing (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, 
        test_size=0.2, 
        random_state=42  # Seed untuk reproducibility
    )
    
    # 4. TRAIN LOGISTIC REGRESSION (CLASSIFICATION)
    logreg = LogisticRegression(max_iter=200)  # Model klasifikasi
    logreg.fit(X_train, y_train)              # Training model
    
    # Evaluasi akurasi pada data test
    y_pred = logreg.predict(X_test)
    print('Logistic Regression accuracy:', accuracy_score(y_test, y_pred))
    
    # 5. TRAIN K-MEANS (CLUSTERING)
    # Clustering seluruh data (tanpa label)
    kmeans = KMeans(n_clusters=2, random_state=42)  # 2 cluster
    kmeans.fit(X_scaled)                            # Training model
    
    # 6. SAVE MODELS & SCALER
    # Menyimpan model dan scaler ke file .joblib untuk digunakan di app.py
    joblib.dump(logreg, 'logreg_model.joblib')
    joblib.dump(kmeans, 'kmeans_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    
    print('Models and scaler saved.')
    return logreg, kmeans, scaler

# Blok eksekusi jika file di-run langsung
if __name__ == '__main__':
    # Jalankan training dengan dataset default
    train_models('Telco-Customer-Churn.csv')