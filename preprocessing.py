# preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(path):
    """
    Memuat dataset dari file CSV.
    Args:
        path (str): Path/lokasi file CSV.
    Returns:
        DataFrame: Data mentah dalam bentuk pandas DataFrame.
    """
    df = pd.read_csv(path)
    return df

def data_cleaning(df):
    """
    Membersihkan data dengan menghapus duplikat, kolom tidak perlu, dan konversi tipe data.
    Args:
        df (DataFrame): Data mentah.
    Returns:
        DataFrame: Data yang sudah dibersihkan.
    """
    # Hapus baris duplikat
    df = df.drop_duplicates()
    
    # Hapus kolom 'customerID' jika ada (karena tidak relevan untuk analisis)
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    
    # Konversi 'TotalCharges' ke numerik (jika ada)
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')  # 'coerce' untuk mengubah invalid values jadi NaN
    
    return df

def handle_missing(df):
    """
    Menangani missing values dengan imputasi.
    - Numerik: diisi median
    - Kategorik: diisi modus
    Args:
        df (DataFrame): Data yang sudah dibersihkan.
    Returns:
        DataFrame: Data tanpa missing values.
    """
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:  # Untuk kolom numerik
            df[col].fillna(df[col].median(), inplace=True)  # Isi missing dengan median
        else:  # Untuk kolom non-numerik
            df[col].fillna(df[col].mode()[0], inplace=True)  # Isi missing dengan modus
    return df

def encode_transform(df):
    """
    Melakukan encoding pada kolom kategorikal.
    - 'Churn' diubah jadi binary (0/1)
    - Kolom kategorik lainnya di-encode dengan LabelEncoder
    Args:
        df (DataFrame): Data yang sudah di-handle missing values.
    Returns:
        DataFrame: Data dengan kolom kategorik ter-encode.
    """
    # Encode kolom target 'Churn' (jika ada)
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})  # Manual mapping untuk target
    
    # Encode kolom kategorik lainnya
    for col in df.select_dtypes(include=['object']).columns:  # Pilih hanya kolom bertipe object/string
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])  # Konversi ke numerik (misal: ['Male','Female'] -> [0,1])
    
    return df

def feature_scaling(df):
    """
    Melakukan standard scaling pada fitur numerik.
    Args:
        df (DataFrame): Data yang sudah ter-encode.
    Returns:
        Tuple: (DataFrame yang sudah di-scale, scaler object untuk transformasi baru)
    """
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df),  # Lakukan scaling (standarisasi: mean=0, std=1)
        columns=df.columns        # Pertahankan nama kolom
    )
    return df_scaled, scaler

def data_preprocessing(path):
    """
    Fungsi utama untuk menjalankan seluruh pipeline preprocessing.
    Args:
        path (str): Path ke file CSV.
    Returns:
        DataFrame: Data yang siap untuk modeling (clean, encoded, scaled).
    """
    # 1. Load data
    df = load_data(path)
    
    # 2. Data cleaning
    df = data_cleaning(df)
    
    # 3. Handle missing values
    df = handle_missing(df)
    
    # 4. Encoding kategorikal
    df = encode_transform(df)
    
    return df  # Note: Scaling dilakukan terpisah di modeling.py untuk menghindari data leakage