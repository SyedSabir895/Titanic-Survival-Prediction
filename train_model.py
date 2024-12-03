import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
data = pd.read_csv(url)

# Select relevant features
data = data[["Pclass", "Sex", "Age", "Fare", "Survived"]]

# Handle missing values
data["Age"].fillna(data["Age"].median(), inplace=True)

# Encode categorical variables
data["Sex"] = data["Sex"].map({"male": 0, "female": 1})

# Define features and target
X = data[["Pclass", "Sex", "Age", "Fare"]]
y = data["Survived"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Save the model
with open("model/titanic_model.pkl", "wb") as f:
    pickle.dump(model, f)
