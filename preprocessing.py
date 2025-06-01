import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(path):
    df = pd.read_csv(path)
    return df

def data_cleaning(df):
    df = df.drop_duplicates()
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    return df

def handle_missing(df):
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
    return df

def encode_transform(df):
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'No':0, 'Yes':1})
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df

def feature_scaling(df):
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_scaled, scaler

def data_preprocessing(path):
    df = load_data(path)
    df = data_cleaning(df)
    df = handle_missing(df)
    df = encode_transform(df)
    return df
