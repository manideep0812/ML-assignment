**Machine Learning Classification – Bank Marketing Dataset**

**Problem Statement -** 
The objective of this project is to build, evaluate, and compare multiple machine learning classification models to predict whether a bank customer will subscribe to a term deposit. The comparison is performed using standard evaluation metrics to identify the most effective model for the given dataset and to understand the strengths and limitations of each approach.

**Dataset Description -**
The dataset used is the Bank Marketing – Additional Dataset from the UCI Machine Learning Repository. 
It contains data collected from direct marketing campaigns of a Portuguese banking institution.

- Dataset size: ~41,000 records  
- Features: 20 input features (numerical and categorical)  
- Target variable: y (binary classification: yes / no) 
This dataset is well-suited for evaluating multiple classification models and performance metrics beyond accuracy.


**Models Used and Evaluation Metrics -**
Six machine learning models were trained and evaluated using the following metrics:
Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

**Comparison Table of Model Performance -**
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.9138 | 0.9385 | 0.6667 | 0.4257 | 0.5196 | 0.4892 |
| Decision Tree | 0.9378 | 0.9658 | 0.7671 | 0.6208 | 0.6863 | 0.6567 |
| kNN | 0.9235 | 0.9568 | 0.7361 | 0.4701 | 0.5737 | 0.5502 |
| Naive Bayes | 0.7429 | 0.8365 | 0.2711 | 0.7982 | 0.4047 | 0.3570 |
| Random Forest | 0.9709 | 0.9890 | 0.9853 | 0.7450 | 0.8485 | 0.8427 |
| XGBoost | 0.9517 | 0.9763 | 0.8298 | 0.7029 | 0.7611 | 0.7376 |


**Model-wise Observations -**
| ML Model Name | Observation about Model Performance |
|--------------|------------------------------------|
| Logistic Regression | Shows stable performance with good AUC but relatively low recall, indicating difficulty in identifying minority-class subscribers. |
| Decision Tree | Captures non-linear patterns effectively and improves recall over Logistic Regression, but may still be prone to overfitting. |
| kNN | Achieves moderate performance but is sensitive to distance calculations and class imbalance, leading to limited recall. |
| Naive Bayes | Achieves very high recall but extremely low precision, resulting in poor overall performance and many false positives. |
| Random Forest | Delivers the best overall performance with excellent accuracy, AUC, F1 score, and MCC, demonstrating strong generalization and robustness. |
| XGBoost | Performs exceptionally well with balanced precision and recall, slightly below Random Forest but still superior to single models. |
