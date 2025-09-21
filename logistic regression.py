import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

#  Load the dataset 
df = pd.read_csv("final dataset.csv")

#  Define input features (X) and target variable (y) 
selected_features = ['has_e330','has_e500','has_e202','has_e322','has_colorant','has_sweetener','has_emulsifier','has_added_sugar','has_oil_and_fat','has_carcinogen','has_allergen','has_cardiovascular_risk_ingredient','number_of_ingredients_norm','number_of_additives_norm','additives_per_ingredient_norm','has_cheese_marker','complex_non_additive_score_norm','salt_and_food_combo',
]

X = df[selected_features]
y = df['Nova Score']

#  Train-test split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#  Initialize and train Logistic Regression 
model = LogisticRegression(max_iter=1000, multi_class='ovr', solver='liblinear')
model.fit(X_train, y_train)

#  Make predictions 
y_pred = model.predict(X_test)

#  Evaluate the model 
print("\n=== Evaluation Metrics ===")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred, average='weighted'):.4f}")
print(f"F1-score:  {f1_score(y_test, y_pred, average='weighted'):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#  Confusion Matrix 
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=[f"NOVA {i}" for i in sorted(y.unique())],
            yticklabels=[f"NOVA {i}" for i in sorted(y.unique())])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Logistic Regression")
plt.tight_layout()
plt.savefig("logreg_confusion_matrix.png")
plt.show()


joblib.dump(model, "final_logreg_model.joblib")
