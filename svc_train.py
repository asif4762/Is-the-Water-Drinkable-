import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

df = pd.read_csv('https://raw.githubusercontent.com/asif4762/Is-the-Water-Drinkable-/refs/heads/main/Datasets/water_potability.csv')
# print(df.head())
target_col = 'Potability'
X = df.drop(target_col, axis=1)
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

num_col = X_train.select_dtypes(include=np.number).columns

num_transform = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transform, num_col)
    ]
)

model = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('classifier', SVC(C=1.0, gamma='scale', kernel='rbf'))
    ]
)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

with open('svc_model.pkl', 'wb') as f:
    pickle.dump(model, f)
    
print('Model saved successfully!')