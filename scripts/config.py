import os
import pandas as pd

# Directory structure
output_dir = os.path.abspath(os.path.join(os.getcwd(), "../results/plots"))
raw_data_file_path = os.path.abspath(os.path.join(os.getcwd(), "../data/raw/creditcard.csv"))
cleaned_data_file_path = os.path.abspath(os.path.join(os.getcwd(), "../data/processed/cleaned_dataset.csv"))
processed_data_dir = os.path.abspath(os.path.join(os.getcwd(), "../data/processed"))
models_dir = os.path.abspath(os.path.join(os.getcwd(), "../models"))
model_file = os.path.join(models_dir, "Logistic Regression.pkl")
 
# Load processed datasets
X_train = pd.read_csv(os.path.join(processed_data_dir, "X_train.csv"))
X_test = pd.read_csv(os.path.join(processed_data_dir, "X_test.csv"))
y_train = pd.read_csv(os.path.join(processed_data_dir, "y_train.csv")).values.ravel()
y_test = pd.read_csv(os.path.join(processed_data_dir, "y_test.csv")).values.ravel()

# Metrics and predictions directories
predictions_dir = os.path.abspath(os.path.join(os.getcwd(), "../results/predictions"))
matrix_dir = os.path.abspath(os.path.join(os.getcwd(), "../results/Matrix/NearMiss"))
metrics_evaluation_dir = os.path.abspath(os.path.join(os.getcwd(), "../results/Evaluation_Metrics/NearMiss"))
metrics_dir = os.path.abspath(os.path.join(os.getcwd(), "../results/metrics"))
metrics_dir_ADASYN = os.path.abspath(os.path.join(os.getcwd(), "../results/metrics/ADASYN"))
metrics_dir_SMOTE = os.path.abspath(os.path.join(os.getcwd(), "../results/metrics/SMOTE"))
metrics_dir_SVM_SMOTE = os.path.abspath(os.path.join(os.getcwd(), "../results/metrics/SVM-SMOTE"))
metrics_dir_Tomek_Links = os.path.abspath(os.path.join(os.getcwd(), "../results/metrics/Tomek Links"))
metrics_dir_NearMiss = os.path.abspath(os.path.join(os.getcwd(), "../results/metrics/NearMiss"))

metrics_file = os.path.join(metrics_dir, "metrics.json")
metrics_file_SMOTE = os.path.join(metrics_dir_SMOTE, "metrics.json")
metrics_file_ADASYN = os.path.join(metrics_dir_ADASYN, "metrics.json")
metrics_file_SVM_SMOTE = os.path.join(metrics_dir_SVM_SMOTE, "metrics.json")
metrics_file_Tomek_Links = os.path.join(metrics_dir_Tomek_Links, "metrics.json")
metrics_file_NearMiss = os.path.join(metrics_dir_NearMiss, "metrics.json")

