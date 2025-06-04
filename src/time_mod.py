import pandas as pd
import time
import joblib
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

# ---- Load and prepare dataset ----
df = pd.read_csv("../data/filtered_2015_onwards.csv")
df['loan_status'] = df['loan_status'].apply(lambda x: 1 if x in ['Charged Off', 'Default'] else 0)

categorical_features = ['term', 'grade', 'sub_grade', 'emp_length', 'home_ownership', 'purpose', 'addr_state']
numeric_features = ['loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti',
                    'delinq_2yrs', 'fico_range_high', 'fico_range_low', 'open_acc',
                    'pub_rec', 'revol_util', 'total_acc']

X = df[numeric_features + categorical_features]
y = df['loan_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# ---- Load preprocessor ----
preprocessor = joblib.load("preprocessor.pkl")
X_train_proc = preprocessor.transform(X_train)
X_test_proc = preprocessor.transform(X_test)

results = []

# ---- LightGBM ----
start_train = time.time()
lgb_train = lgb.Dataset(X_train_proc, label=y_train)
lgb_model = lgb.train({'objective': 'binary'}, lgb_train, num_boost_round=100)
end_train = time.time()

start_pred = time.time()
_ = lgb_model.predict(X_test_proc)
end_pred = time.time()

results.append({
    'Model': 'LightGBM',
    'Train Time (s)': round(end_train - start_train, 2),
    'Predict Time (s)': round(end_pred - start_pred, 2)
})

# ---- XGBoost ----
start_train = time.time()
xgb_model = xgb.train({'objective': 'binary:logistic'},
                      xgb.DMatrix(X_train_proc, label=y_train),
                      num_boost_round=100)
end_train = time.time()

start_pred = time.time()
_ = xgb_model.predict(xgb.DMatrix(X_test_proc))
end_pred = time.time()

results.append({
    'Model': 'XGBoost',
    'Train Time (s)': round(end_train - start_train, 2),
    'Predict Time (s)': round(end_pred - start_pred, 2)
})

# ---- CatBoost ----
X_train_cat = X_train.copy()
X_test_cat = X_test.copy()
for col in categorical_features:
    X_train_cat[col] = X_train_cat[col].fillna("Unknown").astype(str)
    X_test_cat[col] = X_test_cat[col].fillna("Unknown").astype(str)

cat_features_indices = [X_train_cat.columns.get_loc(col) for col in categorical_features]

start_train = time.time()
cat_model = CatBoostClassifier(iterations=100, learning_rate=0.1, verbose=0)
cat_model.fit(X_train_cat, y_train, cat_features=cat_features_indices)
end_train = time.time()

start_pred = time.time()
_ = cat_model.predict(X_test_cat)
end_pred = time.time()

results.append({
    'Model': 'CatBoost',
    'Train Time (s)': round(end_train - start_train, 2),
    'Predict Time (s)': round(end_pred - start_pred, 2)
})

# ---- Print Results ----
print("\nModel Speed Benchmark:")
for r in results:
    print(f"{r['Model']:10} | Train: {r['Train Time (s)']}s | Predict: {r['Predict Time (s)']}s")

