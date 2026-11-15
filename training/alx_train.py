import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from pathlib import Path

HERE = Path(__file__).resolve().parent

print("Loading data...")
df = pd.read_csv(HERE / 'landmark_data.csv')

print(f"Total rows: {len(df)}")
print(f"Rows with NaN: {df.isnull().any(axis=1).sum()}")

df = df.dropna()
print(f"Rows after dropping NaN: {len(df)}")

X = df.drop('label', axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training model...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy}")

print("Saving model...")
joblib.dump(clf, HERE.parent / 'ALX_model_local.joblib')
print("Model saved successfully!")