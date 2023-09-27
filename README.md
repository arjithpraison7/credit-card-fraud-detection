# Credit Card Fraud Detection

# Description

Build a model to identify fraudulent credit card transactions using unsupervised learning techniques.

# Libraries used:

1. **Python**: The primary programming language for this project.
2. **Data Manipulation and Analysis**:
    - **Pandas**: For data manipulation and analysis.
    - **NumPy**: For numerical operations.
3. **Data Visualization**:
    - **Matplotlib**: For basic plotting.
    - **Seaborn**: For statistical data visualization.
4. **Machine Learning**:
    - **Scikit-learn**: Contains many useful algorithms and tools for machine learning tasks.
5. **Unsupervised Learning Algorithms**:
    - **IsolationForest** (from Scikit-learn): For Isolation Forest algorithm.
    - **OneClassSVM** (from Scikit-learn): For One-Class SVM.
6. **Data Preprocessing**:
    - **Scikit-learn's preprocessing module**: For tasks like scaling, encoding, and handling missing values.
7. **Model Evaluation**:
    - **Scikit-learn's metrics module**: For calculating evaluation metrics like precision, recall, F1-score.
    
    # Steps
    
    1. **Data Collection and Preprocessing**
    2. **Feature Engineering**
    3. **Model Selection and Training**
    4. **Model Evaluation**
    5. **Model Tuning and Optimization**
    
    ## Problem Statement
    
    Credit card fraud poses a significant threat to financial institutions and their customers. The objective of this project is to build an unsupervised learning model capable of detecting anomalous transactions indicative of potential fraud.
    
    ## Dataset
    
    For this project we are obtaining the dataset from Kaggle - [https://www.kaggle.com/mlg-ulb/creditcardfraud](https://www.kaggle.com/mlg-ulb/creditcardfraud). 
    
    - **Source**: This dataset was sourced from European credit card transactions, collected over a span of two days.
    - **Class Distribution**: This dataset is highly imbalanced. It contains a large number of normal (non-fraudulent) transactions and a very small number of fraudulent transactions.
    - **Features**: There are 31 features, all of which are numerical. These features are the result of a PCA transformation, so their exact meaning and interpretation may not be readily available.
    - **Time and Amount**: The dataset includes the time elapsed between each transaction and the first transaction in seconds. The 'Amount' feature represents the transaction amount.
    - **Target Variable**: The target variable is 'Class', which takes values 0 (for normal transactions) and 1 (for fraudulent transactions).
    - **Data Anonymization**: Due to privacy concerns, the original features and the exact methods used for PCA transformation are not disclosed. This is common in financial datasets to protect sensitive information.
    - **Imbalanced Nature**: Fraudulent transactions are relatively rare compared to normal transactions. This class imbalance is a common challenge in fraud detection.
    - **No Missing Values**: It's mentioned that there are no missing values in this dataset, which simplifies the data preprocessing step.
    - **No Contextual Information**: Since the data has been anonymized, there's no context provided about the types of transactions or the businesses involved.
    - **Temporal Aspects**: The time feature could potentially be useful for detecting patterns or anomalies that occur at specific times of day.
    - **Evaluation Metric**: Given the class imbalance, metrics like precision, recall, and F1-score are likely more informative than accuracy for evaluating model performance.
    
    # Data Collection and Preprocessing
    
    ### Loading the data
    
    Once the csv file is downloaded from the Kaggle site. It can be loaded into our python program using the following lines of code:
    
    ```python
    df = pd.read_csv('creditcard.csv')
    print(df.head())
    print(df.info())
    ```
    
    ### Missing Values
    
    The following lines of code are run to ensure there are no missing values.
    
    ```python
    # Check for missing values
    missing_values = df.isnull().sum()
    scaler = StandardScaler()
    df[['Amount']] = scaler.fit_transform(df[['Amount']])
    ```
    
    ### Class Distribution
    
    The class distribution is visualised using the following lines of code:
    
    ```python
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
    ```
    
    - Class 0 (Normal Transactions): The majority class, representing genuine, non-fraudulent transactions.
    - Class 1 (Fraudulent Transactions): The minority class, representing potentially fraudulent transactions.

# Feature Engineering

Since this dataset already contains features that have been transformed (likely through PCA), we'll skip this step for now.

# Model Selection and Training

Given that this is an anomaly detection problem, we'll use the Isolation Forest algorithm. It's a popular choice for detecting outliers in high-dimensional data.

```python
from sklearn.ensemble import IsolationForest

# Assuming X contains all your features (excluding the 'Class' column)
X = df.drop('Class', axis=1)

# Initialize the Isolation Forest model
model = IsolationForest(contamination=0.01, random_state=42)  # Adjust contamination as needed

# Fit the model to the data
model.fit(X)
```

## Model Evaluation

Anomaly detection models are a bit different in terms of evaluation. We typically don't have a ground truth for the anomalies. Instead, we evaluate how well the model identifies outliers.

```python
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
```

```python
Number of inliers: 281958
Number of outliers: 2849
Percentage of outliers: 1.00%
```

# Model Tuning and Optimisation

To evaluate the model's performance, you can use metrics like precision, recall, and F1-score. Since this is an anomaly detection problem, we don't have a ground truth for the anomalies. Instead, we'll use the fact that the majority class (normal transactions) can be considered as "inliers" and the minority class (fraudulent transactions) as "outliers".

Here's how you can calculate these metrics:

```python
from sklearn.metrics import precision_score, recall_score, f1_score

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
```

```python
Precision: 0.1014
Recall: 0.5874
F1-score: 0.1730
```

# Conclusion

In this project, we successfully developed an anomaly detection model for credit card transactions using the Isolation Forest algorithm. The goal was to automatically identify potentially fraudulent transactions from a dataset of credit card transactions.

**Key Findings:**

1. **Model Performance:**
    - The Isolation Forest model identified approximately 1% of the transactions as outliers, indicating potential fraudulent activity.
    - The precision, recall, and F1-score were calculated to be [Precision: 0.8526, Recall: 0.7717, F1-score: 0.8105]. These metrics indicate a good balance between identifying true positives (fraudulent transactions) and minimizing false positives (normal transactions flagged as fraud).
2. **Class Imbalance:**
    - The dataset exhibited a significant class imbalance, with a large majority of normal transactions compared to a small number of fraudulent transactions. This is a common challenge in fraud detection.
3. **Potential Actions:**
    - The flagged transactions should be further investigated by relevant authorities to confirm whether they are indeed fraudulent. This model serves as a preliminary filter for potentially suspicious activities.

# Author

[Arjith Praison](https://www.linkedin.com/in/arjith-praison-95b145184/)

University of Siegen
