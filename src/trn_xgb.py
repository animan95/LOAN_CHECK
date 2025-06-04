import joblib
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split

# ---- 1. Load filtered dataset ----
df = pd.read_csv("../data/filtered_2015_onwards.csv")
df['loan_status'] = df['loan_status'].apply(lambda x: 1 if x in ['Charged Off', 'Default'] else 0)

# ---- 2. Define features (MUST match original set used to fit preprocessor) ----
categorical_features = ['term', 'grade', 'sub_grade', 'emp_length', 'home_ownership', 'purpose', 'addr_state']
numeric_features = ['loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti',
                    'delinq_2yrs', 'fico_range_high', 'fico_range_low', 'open_acc',
                    'pub_rec', 'revol_util', 'total_acc']

X = df[numeric_features + categorical_features]
y = df['loan_status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# ---- 3. Load preprocessor and transform data ----
preprocessor = joblib.load('preprocessor.pkl')
X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# DMatrix is XGBoost’s optimized data format
dtrain = xgb.DMatrix(X_train_processed, label=y_train)
dtest = xgb.DMatrix(X_test_processed, label=y_test)

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'eta': 0.1,
    'max_depth': 6,
    'seed': 42
}

# Train model
model = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    evals=[(dtrain, 'train'), (dtest, 'test')],
    early_stopping_rounds=10,
    verbose_eval=False
)

# Predict
y_pred_proba = model.predict(dtest)
y_pred = (y_pred_proba > 0.5).astype(int)

# Evaluate
print("✅ Classification Report:\n", classification_report(y_test, y_pred))
print("🎯 ROC AUC Score:", roc_auc_score(y_test, y_pred_proba))
print("📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


joblib.dump(model, 'xgb_credit_model.pkl')
print("💾 Model saved to xgbcredit_model.pkl")
