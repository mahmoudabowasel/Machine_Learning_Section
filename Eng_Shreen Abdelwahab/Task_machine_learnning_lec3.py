# محمود كمال ابو واصل عوض
import pandas as pd
import numpy as np
from math import log2
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier

data = {
    "Study": ["High", "High", "High", "Low", "Low", "Low", "Low", "High"],
    "Attendance": ["Good", "Good", "Poor", "Good", "Poor", "Poor", "Good", "Poor"],
    "Homework": ["Done", "Missed", "Done", "Done", "Missed", "Done", "Missed", "Done"],
    "Result": ["Pass", "Pass", "Pass", "Pass", "Fail", "Fail", "Fail", "Pass"]
}

df = pd.DataFrame(data)
features = ["Study", "Attendance", "Homework"]
X = pd.get_dummies(df[features])
y = df["Result"]

def entropy(labels):
    probabilities = labels.value_counts(normalize=True)
    return -sum(p * log2(p) for p in probabilities if p > 0)

def gini(labels):
    probabilities = labels.value_counts(normalize=True)
    return 1 - sum(p ** 2 for p in probabilities)

def information_gain(frame, feature, target="Result"):
    parent_entropy = entropy(frame[target])
    weighted_entropy = 0.0
    for value, group in frame.groupby(feature):
        weighted_entropy += len(group) / len(frame) * entropy(group[target])
    return parent_entropy - weighted_entropy

print("Student dataset:")
print(df.to_string(index=False))

print("\nDataset measurements:")
print(f"Number of students: {len(df)}")
print(f"Pass: {(y == 'Pass').sum()}, Fail: {(y == 'Fail').sum()}")
print(f"Entropy: {entropy(y):.3f}")
print(f"Gini impurity: {gini(y):.3f}")

print("\nInformation gain for each feature:")
for feature in features:
    print(f"{feature}: {information_gain(df, feature):.3f}")

tree = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42)
tree.fit(X, y)

forest = RandomForestClassifier(
    n_estimators=100,
    max_features="sqrt",
    bootstrap=True,
    random_state=42
)
forest.fit(X, y)

new_students = pd.DataFrame([
    {"Study": "Low", "Attendance": "Poor", "Homework": "Done"},
    {"Study": "Low", "Attendance": "Good", "Homework": "Done"},
    {"Study": "High", "Attendance": "Poor", "Homework": "Missed"},
    {"Study": "Low", "Attendance": "Good", "Homework": "Missed"}
])

X_new = pd.get_dummies(new_students)
X_new = X_new.reindex(columns=X.columns, fill_value=0)

print("\nDecision tree structure:")
print(export_text(tree, feature_names=list(X.columns)))

print("\nPredictions for new students:")
tree_predictions = tree.predict(X_new)
forest_predictions = forest.predict(X_new)

for index, row in new_students.iterrows():
    print(
        f"Student {index + 1}: Study={row['Study']}, "
        f"Attendance={row['Attendance']}, Homework={row['Homework']} | "
        f"Decision Tree={tree_predictions[index]} | "
        f"Random Forest={forest_predictions[index]}"
    )

print("\nModel training accuracy:")
print(f"Decision Tree: {tree.score(X, y) * 100:.1f}%")
print(f"Random Forest: {forest.score(X, y) * 100:.1f}%")

print("\nRandom Forest feature importance:")
importance = pd.Series(forest.feature_importances_, index=X.columns)
print(importance.sort_values(ascending=False).to_string())

print("\nNote: This is a small educational dataset. Training accuracy on 8 rows does not measure performance on unseen real-world data.")
