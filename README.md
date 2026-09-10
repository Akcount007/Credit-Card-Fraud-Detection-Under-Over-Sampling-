# Credit Card Fraud Detection (Under & Over Sampling)

This project aims to detect fraudulent credit card transactions using machine learning classification algorithms[cite: 5]. Because fraud datasets are heavily skewed, this project specifically focuses on handling imbalanced data using both Undersampling and Oversampling (SMOTE) techniques[cite: 5].

**Tutorial Reference:** [YouTube Video Link](https://youtu.be/0pGXW0bCgHE)[cite: 5]

## 🗄️ Dataset
The dataset utilized is **`creditcard.csv`**[cite: 5]. 
* **Download Link:** [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* The data consists of various anonymized features (`V1` to `V28`), `Time`, `Amount`, and the target `Class`[cite: 5].
* `Class 0` represents normal transactions, and `Class 1` represents fraudulent transactions[cite: 5].

## ⚙️ Data Preprocessing
To prepare the data for the machine learning models, the following steps were taken:
* **Feature Dropping**: The `Time` column was dropped from the dataset as it was not required[cite: 5].
* **Feature Scaling**: The `Amount` column was scaled using `StandardScaler` to ensure uniform feature distribution[cite: 5].
* **Data Cleaning**: Duplicate entries were identified and removed from the dataset[cite: 5].

## ⚖️ Handling Imbalanced Data
A significant challenge in fraud detection is the massive imbalance between normal and fraudulent transactions. Two methods were explored:
* **Undersampling**: A random sample of normal transactions was extracted to match the lower count of fraudulent transactions[cite: 5].
* **Oversampling (SMOTE)**: The `SMOTE` (Synthetic Minority Over-sampling Technique) algorithm from the `imblearn` library was used to synthetically oversample the minority fraud class[cite: 5].

## 🤖 Machine Learning Models
Three classification algorithms from the `scikit-learn` library were trained and evaluated on the processed data:
1. Logistic Regression (`LogisticRegression`)[cite: 5]
2. Decision Tree Classifier (`DecisionTreeClassifier`)[cite: 5]
3. Random Forest Classifier (`RandomForestClassifier`)[cite: 5]

## 📊 Evaluation Metrics & Results
The models were evaluated using Accuracy, Precision, Recall, and F1-Score[cite: 5]. 

* After applying oversampling, the **Random Forest Classifier** emerged as the best-performing model, achieving approximately **99.99% Accuracy**[cite: 5]. 

## 💾 Model Serialization & Inference
* The highest performing Random Forest model was saved using the `joblib` library under the filename `credit_card_model`[cite: 5].
* The saved model can be loaded back into memory to predict whether new transactions are a `"Normal Transcation"` or `"Fraudulent Transcation"`[cite: 5].

## 🛠️ Requirements & Libraries
To run the code, ensure the following Python libraries are installed. You can click the links to view their official documentation:
* [NumPy](https://numpy.org/)[cite: 5]
* [Pandas](https://pandas.pydata.org/)[cite: 5]
* [Seaborn](https://seaborn.pydata.org/)[cite: 5]
* [Scikit-Learn](https://scikit-learn.org/)[cite: 5]
* [Imbalanced-Learn (SMOTE)](https://imbalanced-learn.org/)[cite: 5]
* [Joblib](https://joblib.readthedocs.io/)[cite: 5]
