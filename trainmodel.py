import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from joblib import dump


# ======================================
# THERMAL STRESS MODEL
# ==============================
# ========

print("Loading thermal features...")

thermal_features = np.load("thermal_features.npy")
thermal_labels = np.load("thermal_labels.npy", allow_pickle=True)

print("Encoding stress labels...")

le_stress = LabelEncoder()
y_stress = le_stress.fit_transform(thermal_labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    thermal_features,
    y_stress,
    test_size=0.2,
    random_state=42
)

print("Training thermal stress model...")

svm_stress = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
svm_stress.fit(X_train, y_train)

# Evaluate model
pred = svm_stress.predict(X_test)

acc = accuracy_score(y_test, pred)
prec = precision_score(y_test, pred, average="weighted")
rec = recall_score(y_test, pred, average="weighted")
f1 = f1_score(y_test, pred, average="weighted")

print("\nThermal Stress Model Performance")
print("Accuracy :", round(acc*100,2),"%")
print("Precision:", round(prec,3))
print("Recall   :", round(rec,3))
print("F1 Score :", round(f1,3))

# Save model
dump(svm_stress, "svm_stress.pkl")
dump(le_stress, "label_encoder_stress.pkl")


# ======================================
# RGB DISEASE MODEL
# ======================================

print("\nLoading RGB features...")

rgb_features = np.load("rgb_features.npy")
rgb_labels = np.load("rgb_labels.npy", allow_pickle=True)

print("Encoding disease labels...")

le_disease = LabelEncoder()
y_disease = le_disease.fit_transform(rgb_labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    rgb_features,
    y_disease,
    test_size=0.2,
    random_state=42
)

print("Training RGB disease model...")

svm_disease = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
svm_disease.fit(X_train, y_train)

# Evaluate model
pred = svm_disease.predict(X_test)

acc = accuracy_score(y_test, pred)
prec = precision_score(y_test, pred, average="weighted")
rec = recall_score(y_test, pred, average="weighted")
f1 = f1_score(y_test, pred, average="weighted")

print("\nRGB Disease Model Performance")
print("Accuracy :", round(acc*100,2),"%")
print("Precision:", round(prec,3))
print("Recall   :", round(rec,3))
print("F1 Score :", round(f1,3))

# Save model
dump(svm_disease, "svm_disease.pkl")
dump(le_disease, "label_encoder_disease.pkl")


print("\nTraining completed successfully.")