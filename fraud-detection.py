import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score



df = pd.read_csv('creditcard.csv')
print(df.head())
print(df.info())

# Check for missing values
missing_values = df.isnull().sum()
scaler = StandardScaler()
df[['Amount']] = scaler.fit_transform(df[['Amount']])

# Explore Class Distribuition
class_distribution = df['Class'].value_counts()
print(class_distribution)

# Visualize Class Distribution
plt.figure(figsize=(8, 6))
plt.bar(class_distribution.index, class_distribution.values, color=['blue', 'red'])
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.xticks(class_distribution.index, ['Normal', 'Fraud'])
plt.show()

# Assuming X contains all your features (excluding the 'Class' column)
X = df.drop('Class', axis=1)

# Initialize the Isolation Forest model
model = IsolationForest(contamination=0.01, random_state=42)  # Adjust contamination as needed

# Fit the model to the data
model.fit(X)

# Predict whether each transaction is an outlier (1 for inliers, -1 for outliers)
predictions = model.predict(X)

# Count the number of inliers (1) and outliers (-1)
inliers = (predictions == 1).sum()
outliers = (predictions == -1).sum()

# Calculate the percentage of outliers in the dataset
percentage_outliers = outliers / (inliers + outliers) * 100

print(f'Number of inliers: {inliers}')
print(f'Number of outliers: {outliers}')
print(f'Percentage of outliers: {percentage_outliers:.2f}%')

# Assuming 'predictions' contains the model predictions (-1 for outliers, 1 for inliers)
# Transform predictions to match the labels in the dataset (0 for normal, 1 for fraud)
predictions[predictions == 1] = 0
predictions[predictions == -1] = 1

# True labels (0 for normal, 1 for fraud)
true_labels = df['Class']

# Calculate precision, recall, and F1-score
precision = precision_score(true_labels, predictions)
recall = recall_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions)

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-score: {f1:.4f}')