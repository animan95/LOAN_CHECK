import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
import xgboost as xgb

# Load data
df = pd.read_csv("../data/filtered_2015_onwards.csv")
df['loan_status'] = df['loan_status'].apply(lambda x: 1 if x in ['Charged Off', 'Default'] else 0)

# Define features
categorical_features = ['term', 'grade', 'sub_grade', 'emp_length', 'home_ownership', 'purpose', 'addr_state']
numeric_features = ['loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti',
                    'delinq_2yrs', 'fico_range_high', 'fico_range_low', 'open_acc',
                    'pub_rec', 'revol_util', 'total_acc']
X = df[numeric_features + categorical_features]
y = df['loan_status']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Load preprocessor and transform
preprocessor = joblib.load("preprocessor.pkl")
X_test_proc = preprocessor.transform(X_test)

# Load models
lgbm_model = joblib.load("lgbm_credit_model.pkl")
xgb_model = joblib.load("xgb_credit_model.pkl")
cat_model = joblib.load("ctb_credit_model.pkl")

# Predict
lgbm_preds = lgbm_model.predict(X_test_proc)
xgb_preds = xgb_model.predict(xgb.DMatrix(X_test_proc))
X_test_cat = X_test.copy()
for col in categorical_features:
    X_test_cat[col] = X_test_cat[col].fillna("Unknown").astype(str)
cat_preds = cat_model.predict_proba(X_test_cat)[:, 1]

# Plot ROC curves
fpr_lgb, tpr_lgb, _ = roc_curve(y_test, lgbm_preds)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_preds)
fpr_cat, tpr_cat, _ = roc_curve(y_test, cat_preds)

plt.figure(figsize=(8, 6))
plt.plot(fpr_lgb, tpr_lgb, label=f"LightGBM (AUC = {roc_auc_score(y_test, lgbm_preds):.3f})")
plt.plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC = {roc_auc_score(y_test, xgb_preds):.3f})")
plt.plot(fpr_cat, tpr_cat, label=f"CatBoost (AUC = {roc_auc_score(y_test, cat_preds):.3f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=300)
plt.show()

