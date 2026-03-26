"""
End-to-End Machine Learning Pipeline
Ready Example
Author: Suraj Chaudhary
"""

# =============================
# 1. Import Libraries
# =============================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("🚀 Starting ML Pipeline...\n")

# =============================
# 2. Create Dataset
# =============================
data = {
    "Age": [25, 30, 45, 35, 22, 40, 28, 50, 48, 33, 29, 41],
    "Salary": [50000, 60000, 80000, 65000, 48000, 90000, 52000, 100000, 95000, 62000, 58000, 87000],
    "Experience": [1, 3, 10, 5, 0, 12, 2, 15, 14, 4, 2, 11],
    "Purchased": [0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1]
}

df = pd.DataFrame(data)

print("📊 Dataset Preview:\n")
print(df)

# =============================
# 3. Split Features & Target
# =============================
X = df.drop("Purchased", axis=1)
y = df["Purchased"]

# =============================
# 4. Train Test Split
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n✅ Training Class Distribution:")
print(y_train.value_counts())

# =============================
# 5. Create Pipeline
# =============================
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

# =============================
# 6. Train Model
# =============================
pipeline.fit(X_train, y_train)

# =============================
# 7. Predictions
# =============================
y_pred = pipeline.predict(X_test)

# =============================
# 8. Evaluation
# =============================
print("\n🎯 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📄 Classification Report:\n")
print(classification_report(y_test, y_pred))

# =============================
# 9. Save Model
# =============================
joblib.dump(pipeline, "customer_purchase_model.pkl")
print("\n💾 Model Saved Successfully!")

# =============================
# 10. Load Model
# =============================
loaded_model = joblib.load("customer_purchase_model.pkl")

# =============================
# 11. New Prediction
# =============================
new_customer = [[29, 58000, 2]]

prediction = loaded_model.predict(new_customer)

print("\n🧠 New Customer Prediction:", prediction)

if prediction[0] == 1:
    print("✅ Customer will PURCHASE")
else:
    print("❌ Customer will NOT PURCHASE")

print("\n🏁 Pipeline Completed Successfully!")
