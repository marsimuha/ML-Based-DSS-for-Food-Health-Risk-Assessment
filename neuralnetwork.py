import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("Final Dataset.csv")

# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# Define features and target
selected_features = [
    'has_e330', 'has_e500', 'has_e202', 'has_e322', 'has_colorant', 'has_sweetener',
    'has_emulsifier', 'has_added_sugar', 'has_oil_and_fat', 'has_carcinogen',
    'has_allergen', 'has_cardiovascular_risk_ingredient', 'has_cheese_marker',
    'number_of_ingredients_norm', 'number_of_additives_norm',
    'additives_per_ingredient_norm', 'complex_non_additive_score_norm',
    'salt_and_food_combo'
]

X = df[selected_features]
y = df['nova_score']

# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Initialize and train the neural network model
mlp_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
mlp_model.fit(X_train, y_train)

# Predict
y_pred = mlp_model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}\n")
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=[f"NOVA {i}" for i in sorted(y.unique())], yticklabels=[f"NOVA {i}" for i in sorted(y.unique())])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Neural Network")
plt.tight_layout()
plt.show()

