
# Customer Churn Prediction Using Machine Learning

## Project Overview

This project analyzes telecommunications customer data to identify patterns associated with customer churn and develop machine learning models that predict whether a customer is likely to discontinue their service.

The project combines exploratory data analysis (EDA), data preprocessing, classification algorithms, hyperparameter tuning, and model evaluation to demonstrate how data science can support customer retention strategies.

Three model configurations were evaluated:

- Logistic Regression
- Random Forest Classifier
- Tuned Random Forest Classifier

**The highest test accuracy was 80.62% using a tuned Random Forest model, while Logistic Regression achieved the highest F1-score of 60.40%.**

## Dataset

**Dataset:** IBM Telco Customer Churn

The dataset contains 7,043 customer records with information about demographics, subscriptions, account history, payment methods, and customer churn.

Key variables include:

- `tenure` — Number of months a customer has remained with the company
- `MonthlyCharges` — Monthly service charges
- `TotalCharges` — Total amount charged to a customer
- `Contract` — Month-to-month, one-year, or two-year contract
- `InternetService` — DSL, fiber optic, or no internet service
- `PaymentMethod` — Customer payment method
- `Churn` — Whether the customer discontinued service

The target variable, `Churn`, was converted into binary values:

- `1` = Customer churned
- `0` = Customer remained

## Technologies Used

**Programming Language:** Python

**Libraries:**

- Pandas — Data manipulation and preprocessing
- Matplotlib — Data visualization
- Seaborn — Statistical visualizations
- scikit-learn — Machine learning and model evaluation
- pathlib — File and directory management (Python standard library)

**Development Environment:** Spyder

## Data Cleaning and Preprocessing

The dataset was inspected for missing values, duplicate records, incorrect data types, and categorical variables.

The following preprocessing steps were performed:

1. Identified missing and blank observations.
2. Checked for duplicate customer IDs and duplicate records.
3. Converted `TotalCharges` from string values to numerical values.
4. Identified 11 customers with missing `TotalCharges` values and zero tenure.
5. Replaced these missing charges with zero.
6. Converted the target variable into binary values.
7. Applied one-hot encoding to categorical features.
8. Split the dataset into 80% training and 20% testing data using a stratified split.
9. Standardized `tenure`, `MonthlyCharges`, and `TotalCharges` for Logistic Regression.

Feature scaling was fitted using only the training data to avoid test-set information leakage from scaling.

## Exploratory Data Analysis

Exploratory data analysis investigated relationships between customer characteristics and churn across five categories:

- Contract type
- Customer tenure
- Monthly charges
- Internet service
- Payment method

### Key Findings

**1. Contract Type**

Month-to-month customers demonstrated the highest churn rate, approximately 42.7%, compared with approximately 11.3% for one-year contracts and 2.8% for two-year contracts.

This suggests that customers with shorter contractual commitments are associated with higher churn rates.

**2. Customer Tenure**

Customers with 0–12 months of tenure demonstrated the highest churn rate, approximately 47%. Churn decreased among customers with longer account histories, with long-term customers demonstrating approximately 10% churn.

**3. Monthly Charges**

Customer churn varied across monthly charge categories. The analysis identified a middle-range monthly charge group as having the highest churn rate, suggesting that the relationship between monthly charges and customer churn is not necessarily linear.

**4. Internet Service**

Fiber optic customers demonstrated the highest churn rate at approximately 41.9%. DSL customers demonstrated approximately 19.0% churn, while customers without internet service demonstrated approximately 7.4%.

**5. Payment Method**

Customers using electronic checks demonstrated the highest churn rate at approximately 45.3%. Customers using automatic credit card payments demonstrated a substantially lower churn rate of approximately 15.2%.

These findings describe associations and do not establish that the customer characteristics directly cause churn.

## Machine Learning Methodology

### Logistic Regression

A Logistic Regression classifier was trained to predict customer churn using encoded customer characteristics. Numerical variables were standardized using `StandardScaler`, and performance was evaluated on the held-out testing dataset.

### Random Forest

A Random Forest classifier was trained using the same training and testing split. Random Forest can capture nonlinear relationships and interactions between customer characteristics.

### Hyperparameter Tuning

`GridSearchCV` was used to optimize Random Forest hyperparameters with five-fold cross-validation.

The following parameters were evaluated:

- `n_estimators`: 100, 200
- `max_depth`: 5, 10, None
- `min_samples_split`: 2, 5, 10

The grid search evaluated 18 hyperparameter combinations across five cross-validation folds.

**Best hyperparameters:**

| Hyperparameter | Selected Value |
|---|---|
| `n_estimators` | 200 |
| `max_depth` | 10 |
| `min_samples_split` | 10 |

**Best cross-validation F1-score:** 57.81%

## Model Evaluation

Model performance was evaluated using accuracy, precision, recall, F1-score, and a confusion matrix.

### Model Performance Comparison

| Metric | Logistic Regression | Random Forest | Tuned Random Forest |
|---|---:|---:|---:|
| Accuracy | 80.55% | 78.71% | **80.62%** |
| Precision | 65.72% | 62.67% | **67.35%** |
| Recall | **55.88%** | 48.93% | 52.41% |
| F1-score | **60.40%** | 54.95% | 58.95% |

The tuned Random Forest achieved the highest accuracy and precision. Logistic Regression achieved the highest recall and F1-score, indicating a better balance between identifying customers who churned and avoiding incorrect churn predictions.

Because the testing dataset was examined during model development, these comparisons should be interpreted as exploratory rather than as an unbiased estimate of model-selection performance.

### Logistic Regression Confusion Matrix

| | Predicted No Churn | Predicted Churn |
|---|---:|---:|
| Actual No Churn | 926 | 109 |
| Actual Churn | 165 | 209 |

The Logistic Regression model correctly identified 209 customers who churned but failed to identify 165 customers who discontinued service. These false negatives are particularly relevant for retention efforts because they represent customers the company might fail to identify as being at risk of leaving.

## Model Interpretation

Logistic Regression coefficients were examined to identify customer characteristics associated with predicted churn.

### Features Associated With Higher Churn Odds

- `TotalCharges`
- `StreamingTV_Yes`
- `StreamingMovies_Yes`
- `PaymentMethod_Electronic check`
- `OnlineSecurity_No`
- `SeniorCitizen`

### Features Associated With Lower Churn Odds

- `tenure`
- `Contract_Two year`
- `InternetService_DSL`
- `MonthlyCharges`
- `PaperlessBilling_No`

Positive coefficients indicate higher predicted churn odds, while negative coefficients indicate lower predicted churn odds, holding other model features constant. Coefficient interpretation requires caution because several numerical variables are correlated and categorical features were represented using one-hot encoding.

## Business Recommendations

**1. Focus on new customers**

Customers within their first 12 months demonstrated the highest churn rates. Companies could prioritize onboarding support, customer satisfaction monitoring, and early retention programs.

**2. Investigate month-to-month contract churn**

Month-to-month customers demonstrated substantially higher churn rates than customers with longer contracts. Companies could investigate whether contract incentives or loyalty benefits improve retention.

**3. Evaluate fiber optic customer satisfaction**

Fiber optic customers demonstrated elevated churn rates. Further investigation could examine service quality, pricing, customer expectations, and competitive alternatives.

**4. Investigate payment method differences**

Electronic check customers demonstrated higher churn rates than customers using automatic payment methods. Additional analysis could determine whether payment convenience, customer characteristics, or other factors explain this association.

**5. Use predictive models to prioritize retention efforts**

Logistic Regression could help identify customers with elevated predicted churn risk and support targeted outreach, although real-world deployment would require additional validation and monitoring.

## Project Outputs

The project generates visualizations and analytical outputs in the `outputs/` directory.

**Exploratory Data Analysis:**

- `contract_churn.png`
- `tenure_churn.png`
- `monthly_charge_churn.png`
- `internet_rates_churn.png`
- `payment_rates_churn.png`

**Model Evaluation:**

- `prediction_heatmap.png`

**Analytical Outputs:**

- `model_comparison.csv`
- `logistic_regression_coefficients.csv`

## How to Run

1. Clone or download the repository.
2. Install the required Python libraries using `python -m pip install -r requirements.txt`.
3. Place `Telco-Customer-Churn.csv` in the `data/` directory.
4. Run `python src/customer_churn_project3_final.py` from the project root directory.
5. Review console output and generated files in `outputs/`.

The project uses paths relative to the Python script to locate data and save outputs.

Hyperparameter tuning can be enabled when additional model optimization is needed. Routine executions can use the previously selected Random Forest hyperparameters to reduce runtime.

## Limitations and Future Improvements

- Compare additional classification algorithms, including Gradient Boosting.
- Use cross-validation to tune Logistic Regression.
- Evaluate alternative feature engineering techniques.
- Investigate decision threshold optimization to improve churn recall.
- Use a separate validation strategy for model selection and reserve an untouched test set for final evaluation.
- Develop an interactive dashboard for exploring customer churn patterns.
- Test the models using newer or independently collected customer data.

## Conclusion

This project demonstrates an end-to-end machine learning workflow, including data cleaning, exploratory data analysis, feature preprocessing, classification, hyperparameter tuning, and model interpretation.

The results demonstrate that more complex machine learning models do not necessarily outperform simpler alternatives. Although the tuned Random Forest achieved slightly higher accuracy, Logistic Regression produced the highest recall and F1-score, making it a useful baseline for identifying customers at risk of churn.

The project also highlights how predictive analytics can support business decisions by identifying customer segments that may benefit from additional retention efforts.
