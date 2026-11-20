#Author: etrykk
#Date: 11/18/2026
#Customer Churn Prediction

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

#Change to True when you want to run hyperparameter tuning again
RUN_GRID_SEARCH = False


def load_data():
    #Read dataset
    telco_data = pd.read_csv(DATA_DIR / "Telco-Customer-Churn.csv")

    #Print statements used to identify variables, inspect the data,
    #determine dataset size, check data types for necessary conversions,
    #and display descriptive statistics.
    print(telco_data.columns)
    print(telco_data.head(), telco_data.shape, telco_data.dtypes, telco_data.describe())

    return telco_data


def clean_data(telco_data):
    #checks for values in data that return NaN
    missing_values = telco_data.isnull().sum()
    #counts blank strings
    blank_total_charges = telco_data["TotalCharges"].str.strip().eq("").sum()
    #count number of duplicated IDS
    duplicate_ID = telco_data["customerID"].duplicated().sum()
    #count number of duplicated rows
    duplicate_rows = telco_data.duplicated().sum()
    #Change Total Charges from str to numerical val
    telco_data["TotalCharges"] = pd.to_numeric(telco_data["TotalCharges"], errors="coerce")
    empty_charges = telco_data["TotalCharges"].isnull().sum()

    #set missing customers to dataset where total charges is null
    missing_customers = telco_data[telco_data["TotalCharges"].isnull()]
    #set new customers to missing customers who tenure is 0
    new_customers = missing_customers["tenure"].eq(0).sum()
    #check that missing charges belong to customers with zero tenure
    if empty_charges != new_customers:
        raise ValueError("Some missing TotalCharges belong to customers with nonzero tenure")
    #fill missing data with 0
    telco_data["TotalCharges"] = telco_data["TotalCharges"].fillna(0)
    #check that data was stored correctly
    remaining_missing = telco_data["TotalCharges"].isnull().sum()
    print(remaining_missing)

    print(missing_values)
    print(blank_total_charges)
    print(duplicate_ID)
    print(duplicate_rows)
    print(empty_charges)
    print(telco_data["TotalCharges"].dtype)
    print(new_customers)

    return telco_data


def inspect_categories(cleaned_data):
    #Separate categorical columns from dataset
    categorical_columns = cleaned_data.select_dtypes(include=["object", "str"]).columns

    #print unique values from column
    for column in categorical_columns:
        print(column, cleaned_data[column].unique())

    #count the number of churns and the percentage within the cleaned dataset
    churn_counts = cleaned_data["Churn"].value_counts()
    churn_percentage = cleaned_data["Churn"].value_counts(normalize=True) * 100

    print(churn_counts)
    print(churn_percentage)


def analyze_churn(cleaned_data):
    #Separates each contract type with their churn data
    contract_data = cleaned_data.groupby("Contract")["Churn"]
    #Calculates percentage for churn within each contract type
    contract_churn_percentage = contract_data.value_counts(normalize=True) * 100
    #Extracts the churn percentage for each contract type
    contract_churn_rates = contract_churn_percentage.xs("Yes", level="Churn")
    #calculates highest and lowest churn contracts
    contract_highest_churn = contract_churn_rates.idxmax()
    contract_lowest_churn = contract_churn_rates.idxmin()
    print(contract_churn_rates)
    print(contract_highest_churn, contract_churn_rates.max())
    print(contract_lowest_churn, contract_churn_rates.min())

    #group customers by tenure
    cleaned_data["TenureGroup"] = pd.cut(cleaned_data["tenure"],
                                          bins=[-1, 12, 24, 48, 72],
                                          labels=["New Customers",
                                                  "Developing Customers",
                                                  "Established Customers",
                                                  "Long Term Customers"])

    print(cleaned_data[["tenure", "TenureGroup"]].head(10))
    tenure_group = cleaned_data.groupby("TenureGroup", observed=True)["Churn"]
    tenure_churn_percentage = tenure_group.value_counts(normalize=True) * 100
    tenure_churn_rates = tenure_churn_percentage.xs("Yes", level="Churn")
    tenure_highest_churn = tenure_churn_rates.idxmax()
    tenure_lowest_churn = tenure_churn_rates.idxmin()
    print(tenure_churn_rates)
    print(tenure_highest_churn, tenure_churn_rates.max())
    print(tenure_lowest_churn, tenure_churn_rates.min())

    #calculate descriptive statistics for monthly charges
    monthly_charges = cleaned_data["MonthlyCharges"]
    max_monthly_charge = monthly_charges.max()
    min_monthly_charge = monthly_charges.min()
    mean_monthly_charge = monthly_charges.mean()
    median_monthly_charge = monthly_charges.median()

    #group monthly charges into four categories
    cleaned_data["MonthlyChargeGroup"] = pd.qcut(x=monthly_charges,
                                                   q=4,
                                                   labels=["Low",
                                                           "Medium-Low",
                                                           "Medium-High",
                                                           "High"])

    print(cleaned_data[["MonthlyCharges", "MonthlyChargeGroup"]].head(10))
    monthly_group = cleaned_data.groupby("MonthlyChargeGroup", observed=True)["Churn"]
    monthly_churn_percentage = monthly_group.value_counts(normalize=True) * 100
    monthly_churn_rates = monthly_churn_percentage.xs("Yes", level="Churn")
    monthly_highest_churn = monthly_churn_rates.idxmax()
    monthly_lowest_churn = monthly_churn_rates.idxmin()

    print(f"Maximum Monthly Charge: ${max_monthly_charge:.2f}")
    print(f"Minimum Monthly Charge: ${min_monthly_charge:.2f}")
    print(f"Mean Monthly Charge: ${mean_monthly_charge:.2f}")
    print(f"Median Monthly Charge: ${median_monthly_charge:.2f}")
    print(monthly_churn_rates)
    print(monthly_highest_churn, monthly_churn_rates.max())
    print(monthly_lowest_churn, monthly_churn_rates.min())

    #calculate churn percentages by internet service
    internet_service_group = cleaned_data.groupby("InternetService")["Churn"]
    internet_service_churn_percentage = internet_service_group.value_counts(normalize=True) * 100
    internet_service_churn_rates = internet_service_churn_percentage.xs("Yes", level="Churn")
    internet_service_highest_churn = internet_service_churn_rates.idxmax()
    internet_service_lowest_churn = internet_service_churn_rates.idxmin()

    print(internet_service_churn_rates)
    print(internet_service_highest_churn, internet_service_churn_rates.max())
    print(internet_service_lowest_churn, internet_service_churn_rates.min())

    #calculate churn percentages by payment method
    payment_method_group = cleaned_data.groupby("PaymentMethod")["Churn"]
    payment_method_churn_percentage = payment_method_group.value_counts(normalize=True) * 100
    payment_method_churn_rates = payment_method_churn_percentage.xs("Yes", level="Churn")
    payment_method_highest_churn = payment_method_churn_rates.idxmax()
    payment_method_lowest_churn = payment_method_churn_rates.idxmin()

    print(payment_method_churn_rates)
    print(payment_method_highest_churn, payment_method_churn_rates.max())
    print(payment_method_lowest_churn, payment_method_churn_rates.min())

    return {
        "Contract": contract_churn_rates,
        "Tenure": tenure_churn_rates,
        "MonthlyCharges": monthly_churn_rates,
        "InternetService": internet_service_churn_rates,
        "PaymentMethod": payment_method_churn_rates
    }


def visualize_churn(churn_results):
    #Generate and save churn visualizations
    contract_rates = churn_results["Contract"]
    plt.figure()
    sns.barplot(x=contract_rates.index, y=contract_rates.values)
    plt.xlabel("Contract Type")
    plt.ylabel("Churn Rate %")
    plt.title("Contract Types and Churn Percentage")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "contract_churn.png")
    plt.show()
    plt.close()

    tenure_rates = churn_results["Tenure"]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=tenure_rates.index, y=tenure_rates.values)
    plt.xlabel("Customer Tenure Group")
    plt.ylabel("Churn Rate %")
    plt.title("Customer Tenure Group and Churn Percentage")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tenure_churn.png")
    plt.show()
    plt.close()

    monthly_rates = churn_results["MonthlyCharges"]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=monthly_rates.index, y=monthly_rates.values)
    plt.xlabel("Monthly Charge Group")
    plt.ylabel("Churn Rate %")
    plt.title("Monthly Group and Churn Percentage")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "monthly_charge_churn.png")
    plt.show()
    plt.close()

    internet_rates = churn_results["InternetService"]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=internet_rates.index, y=internet_rates.values)
    plt.xlabel("Internet Service Type")
    plt.ylabel("Churn Rate %")
    plt.title("Internet Service Type and Churn Percentage")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "internet_rates_churn.png")
    plt.show()
    plt.close()

    payment_rates = churn_results["PaymentMethod"]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=payment_rates.index, y=payment_rates.values)
    plt.xlabel("Payment Method Type")
    plt.ylabel("Churn Rate %")
    plt.title("Payment Method Type and Churn Percentage")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "payment_rates_churn.png")
    plt.show()
    plt.close()


def prepare_model_data(cleaned_data):
    #convert churn to numerical values
    y = cleaned_data["Churn"]
    y = y.map({"Yes": 1, "No": 0})
    #remove customer ID, target variable, and grouped columns
    X = cleaned_data.drop(columns=["customerID", "Churn", "TenureGroup", "MonthlyChargeGroup"])

    #convert categorical features to numerical values
    X_encoded = pd.get_dummies(X, dtype=int)
    print(X_encoded.dtypes)

    return X_encoded, y


def split_data(X, y):
    #split dataset into training and testing groups
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    #train logistic regression model
    model = LogisticRegression(max_iter=10000)
    trained_model = model.fit(X_train, y_train)
    return trained_model


def make_predictions(trained_model, X_test):
    #predict churn using the testing dataset
    y_pred = trained_model.predict(X_test)
    return y_pred


def evaluate_model(y_test, y_pred):
    #calculate accuracy, precision, recall, and F1 score
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1


def create_confusion_matrix(y_test, y_pred):
    #calculate and visualize actual vs predicted values
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Values")
    plt.ylabel("Actual Values")
    plt.title("Actual VS. Predicted Values")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "prediction_heatmap.png")
    plt.show()
    plt.close()


def scale_features(X_train, X_test):
    #create copies of training and testing data
    X_train_copy = X_train.copy()
    X_test_copy = X_test.copy()
    #identify numerical columns to scale
    numerical_columns = ["tenure", "MonthlyCharges", "TotalCharges"]
    standard_scaler = StandardScaler()
    #fit scaler on training data and apply to testing data
    X_train_copy[numerical_columns] = standard_scaler.fit_transform(X_train_copy[numerical_columns])
    X_test_copy[numerical_columns] = standard_scaler.transform(X_test_copy[numerical_columns])
    return X_train_copy, X_test_copy


def analyze_feature_importance(trained_model, X_train):
    #get logistic regression coefficients and feature names
    coefficients = trained_model.coef_[0]
    X_train_columns = X_train.columns
    #create dataframe of coefficients
    df = pd.DataFrame({"Feature": X_train_columns, "Coefficient": coefficients})
    #sort coefficients from highest to lowest
    df = df.sort_values(by="Coefficient", ascending=False)
    print(df.head(10))
    print(df.tail(10))
    #save coefficient results
    df.to_csv(OUTPUT_DIR / "logistic_regression_coefficients.csv", index=False)
    return df


def train_random_forest(X_train, y_train):
    #train original random forest model
    forest = RandomForestClassifier(random_state=42)
    trained_forest = forest.fit(X_train, y_train)
    return trained_forest


def tune_random_forest(X_train, y_train):
    #test different random forest settings using 5 fold cross validation
    forest = RandomForestClassifier(random_state=42)
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5, 10]
    }
    grid_model = GridSearchCV(estimator=forest, param_grid=param_grid,
                              cv=5, scoring="f1", n_jobs=-1)
    trained_grid = grid_model.fit(X_train, y_train)
    print(trained_grid.best_params_)
    print(trained_grid.best_score_)
    print(trained_grid.best_estimator_)
    return trained_grid.best_params_, trained_grid.best_score_, trained_grid.best_estimator_


def train_tuned_forest(X_train, y_train):
    #train forest using the best settings found in the previous grid search
    forest = RandomForestClassifier(n_estimators=200, max_depth=10,
                                    min_samples_split=10, random_state=42)
    trained_forest = forest.fit(X_train, y_train)
    return trained_forest


def save_model_results(results):
    #create dataframe comparing all three models
    results_df = pd.DataFrame(results)
    print("\n--- MODEL COMPARISON ---")
    print(results_df.to_string(index=False))
    #save model comparison to CSV
    results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    return results_df


def main():
    #load, clean, and inspect customer data
    telco_data = load_data()
    cleaned_data = clean_data(telco_data)
    inspect_categories(cleaned_data)

    #analyze churn patterns and save visualizations
    churn_results = analyze_churn(cleaned_data)
    visualize_churn(churn_results)

    #prepare features and split into training and testing data
    X, y = prepare_model_data(cleaned_data)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    #train and evaluate logistic regression
    trained_model = train_model(X_train_scaled, y_train)
    y_pred = make_predictions(trained_model, X_test_scaled)
    accuracy, precision, recall, f1 = evaluate_model(y_test, y_pred)
    print("\n--- Logistic Regression Results ---")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1 Score: {f1:.2%}")
    create_confusion_matrix(y_test, y_pred)
    analyze_feature_importance(trained_model, X_train_scaled)
    results = [{"Model": "Logistic Regression", "Accuracy": accuracy,
                "Precision": precision, "Recall": recall, "F1 Score": f1}]

    #train and evaluate original random forest
    trained_forest = train_random_forest(X_train, y_train)
    y_pred_forest = make_predictions(trained_forest, X_test)
    accuracy, precision, recall, f1 = evaluate_model(y_test, y_pred_forest)
    print("\n--- Random Forest Results ---")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1 Score: {f1:.2%}")
    results.append({"Model": "Random Forest", "Accuracy": accuracy,
                    "Precision": precision, "Recall": recall, "F1 Score": f1})

    #run grid search only when requested
    if RUN_GRID_SEARCH:
        best_params, best_score, best_forest = tune_random_forest(X_train, y_train)
        print(f"Best Cross Validation F1 Score: {best_score:.2%}")
    else:
        #use best settings found in previous grid search to save time
        best_forest = train_tuned_forest(X_train, y_train)

    #evaluate tuned random forest
    y_pred_tuned = make_predictions(best_forest, X_test)
    accuracy, precision, recall, f1 = evaluate_model(y_test, y_pred_tuned)
    print("\n--- Tuned Random Forest Results ---")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1 Score: {f1:.2%}")
    results.append({"Model": "Tuned Random Forest", "Accuracy": accuracy,
                    "Precision": precision, "Recall": recall, "F1 Score": f1})

    #print and save final model comparison
    save_model_results(results)


if __name__ == "__main__":
    main()
