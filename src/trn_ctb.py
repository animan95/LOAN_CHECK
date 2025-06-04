from catboost import CatBoostClassifier, Pool
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

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

# Fill NaNs in categorical columns with a placeholder and convert to string
for col in categorical_features:
    X_train[col] = X_train[col].fillna("Unknown").astype(str)
    X_test[col] = X_test[col].fillna("Unknown").astype(str)


# Indices of categorical features
cat_features_indices = [X.columns.get_loc(col) for col in categorical_features]

# Create CatBoost Pool objects
train_pool = Pool(X_train, label=y_train, cat_features=cat_features_indices)
test_pool = Pool(X_test, label=y_test, cat_features=cat_features_indices)

# Train model
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=42,
    early_stopping_rounds=20,
    verbose=False
)
model.fit(train_pool, eval_set=test_pool)

# Predict
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba > 0.5).astype(int)

# Evaluate
print("✅ Classification Report:\n", classification_report(y_test, y_pred))
print("🎯 ROC AUC Score:", roc_auc_score(y_test, y_pred_proba))
print("📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

joblib.dump(model, 'ctb_credit_model.pkl')
print("💾 Model saved to ctb_credit_model.pkl")
