import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import joblib
import os

DATA_PATH = "data/processed/vehicle_features_reduced.csv"
MODEL_DIR = "models"

def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop("label", axis=1)
    y = df["label"]
    return X, y

def train_models(X_train, y_train):
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)

    knn_model = KNeighborsClassifier(n_neighbors=18)
    knn_model.fit(X_train, y_train)

    return nb_model, knn_model

def evaluate_model(model, X_test, y_test, name="Model"):
    predictions = model.predict(X_test)
    print(f"\n{name} Evaluation Report:\n")
    print(classification_report(y_test, predictions))

def main():
    print("Loading data...")
    X, y = load_data()

    print("Encoding labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training models...")
    nb_model, knn_model = train_models(X_train_scaled, y_train)

    print("Evaluating models...")
    evaluate_model(nb_model, X_test_scaled, y_test, name="Naive Bayes")
    evaluate_model(knn_model, X_test_scaled, y_test, name="K-Nearest Neighbors")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(nb_model, os.path.join(MODEL_DIR, "naive_bayes_model.pkl"))
    joblib.dump(knn_model, os.path.join(MODEL_DIR, "knn_model.pkl"))
    joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

if __name__ == "__main__":
    main()
