import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Create sample dataset
np.random.seed(42)

data = {
    "PurchaseHistory": np.random.randint(1, 50, 100),
    "AppUsageHours": np.random.randint(1, 10, 100),
    "EngagementScore": np.random.randint(1, 100, 100),
    "Churn": np.random.randint(0, 2, 100)
}

# Create DataFrame
df = pd.DataFrame(data)

# Display first 5 rows
print("First 5 rows of dataset:")
print(df.head())

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Select features and target
X = df[["PurchaseHistory", "AppUsageHours", "EngagementScore"]]
y = df["Churn"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predict
prediction = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, prediction)

# Display result
print("\n==============================")
print("RESULT")
print("==============================")
print("Random Forest Accuracy:", accuracy)

print("\nTask Completed Successfully!")
