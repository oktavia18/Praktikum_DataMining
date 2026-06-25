import os
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Pastikan folder dataset ada
os.makedirs("dataset", exist_ok=True)
dataset_path = "dataset/student-mat.csv"

# Buat dataset secara otomatis jika belum ada
if not os.path.exists(dataset_path):
    print("Generating dataset...")
    X_raw, y_raw = make_classification(
        n_samples=6000, 
        n_features=4, 
        n_informative=4, 
        n_redundant=0, 
        random_state=42, 
        class_sep=2.0
    )
    feature_names = ['G1', 'G2', 'studytime', 'absences']
    df = pd.DataFrame(X_raw, columns=feature_names)
    df['G3'] = y_raw
    df.to_csv(dataset_path, sep=";", index=False)
    print(f"Dataset berhasil digenerate di {dataset_path}")
else:
    print(f"Dataset sudah ada di {dataset_path}")

# Load dataset
df = pd.read_csv(dataset_path, sep=";")

# Fitur
X = df[['G1', 'G2', 'studytime', 'absences']]

# Target
y = df['G3']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Model Classifier untuk evaluasi akurasi
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Prediksi
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluasi Akurasi
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print("Akurasi Train :", round(train_acc, 3))
print("Akurasi Test  :", round(test_acc, 3))

# Simpan model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model_regresi.pkl")
print("Model berhasil disimpan!")