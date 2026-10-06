import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load data
data = pd.read_csv("train.csv")

# 2. Fill missing values
data["Age"] = data["Age"].fillna(data["Age"].median())
data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])

# 3. Select features
X = data[[
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked"
]]

# 4. Target
y = data["Survived"]

# 5. Encode categorical data
X = pd.get_dummies(
    X,
    columns=["Sex", "Embarked"],
    dtype=int
)

# 6. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
from sklearn.linear_model import LogisticRegression

# 7. Create the model
model = LogisticRegression(max_iter=1000)

# 8. Train the model
model.fit(X_train, y_train)

print("Model training completed!")
# 9. Make predictions on test data
y_pred = model.predict(X_test)

print("First 10 predictions:")
print(y_pred[:10])
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)










