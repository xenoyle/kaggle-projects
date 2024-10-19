import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

train_data = pd.read_csv("./train.csv")
test_data = pd.read_csv("./test.csv")




print(train_data.head())
print(test_data.head())

# Function to preprocess data
def preprocess_data(df):
    # Handle 'Age' column
    age_imputer = SimpleImputer(strategy='median')
    df['Age'] = age_imputer.fit_transform(df[['Age']])

    # Handle 'Embarked' column
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    return df

train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)

y = train_data["Survived"]

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

X_test = X_test.reindex(columns=X.columns, fill_value=0)

model = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(random_state=1, max_iter=300))
])
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submissionMLPNeuralNetwork.csv', index=False)
print("Your submission was successfully saved!")

print("\nMissing values after preprocessing:")
print(train_data[features].isnull().sum())
print("\nUnique values in 'Embarked' after preprocessing:", train_data['Embarked'].unique())
print("Age statistics after preprocessing:")
print(train_data['Age'].describe())

