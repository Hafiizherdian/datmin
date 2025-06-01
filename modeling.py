import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from preprocessing import data_preprocessing, feature_scaling

def train_models(data_path):
    df = data_preprocessing(data_path)
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_scaled, scaler = feature_scaling(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    logreg = LogisticRegression(max_iter=200)
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    print('Logistic Regression accuracy:', accuracy_score(y_test, y_pred))
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X_scaled)
    joblib.dump(logreg, 'logreg_model.joblib')
    joblib.dump(kmeans, 'kmeans_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    print('Models and scaler saved.')
    return logreg, kmeans, scaler

if __name__ == '__main__':
    train_models('Telco-Customer-Churn.csv')
