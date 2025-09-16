from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

# Load your final dataset
df = pd.read_csv("Final Dataset.csv")
X = df.drop(columns=["Nova Score", "Product Code"])
y = df["Nova Score"]

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Save model
joblib.dump(model, "logreg_model.joblib")
