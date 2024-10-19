import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

train_data = pd.read_csv("./train.csv")
test_data = pd.read_csv("./test.csv")

print(train_data.head())
print(test_data.head())

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

X_test = X_test.reindex(columns=X.columns, fill_value=0)

model = LogisticRegression(random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submissionLogisticRegression.csv', index=False)
print("Your submission was successfully saved!")