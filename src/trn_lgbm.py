import joblib
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

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

#To fit to preprocessing pipeline
preprocessor = joblib.load('preprocessor.pkl')
X_tr_pr = preprocessor.fit_transform(X_train)
X_tst_pr = preprocessor.transform(X_test)

#LGBM dataset
lgb_tr = lgb.Dataset(X_tr_pr, label=y_train)
lgb_tst = lgb.Dataset(X_tst_pr, label=y_test, reference=lgb_tr)

#params

params = {'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt', 'verbosity': -1, 'random_state': 42}

#Training

model = lgb.train(params, lgb_tr, valid_sets=[lgb_tr, lgb_tst], num_boost_round=100)

# Prediction

y_pred_prob = model.predict(X_tst_pr)
y_pred = (y_pred_prob > 0.5).astype(int)

print("Classification report:\n", classification_report(y_test, y_pred))
print("ROC-AUC score:", roc_auc_score(y_test, y_pred_prob))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

joblib.dump(model, 'lgbm_credit_model.pkl')
print("💾 Model saved to lgbm_credit_model.pkl")
