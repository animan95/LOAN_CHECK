import shap
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("../data/filtered_2015_onwards.csv")
df['loan_status'] = df['loan_status'].apply(lambda x: 1 if x in ['Charged Off', 'Default'] else 0)

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

# Load LightGBM model
lgbm_model = joblib.load("lgbm_credit_model.pkl")

# Explain with TreeExplainer (CPU-friendly)
explainer = shap.TreeExplainer(lgbm_model)
shap_values = explainer.shap_values(X_test_proc[:500])  # limit to 500 rows for speed

shap_dense = shap_values.toarray() if hasattr(shap_values, "toarray") else shap_values

# Plot summary
shap.summary_plot(
    shap_dense, 
    features=X_test_proc[:500], 
    feature_names=preprocessor.get_feature_names_out()
)


plt.tight_layout()
plt.savefig("shap_summ_plt.png", dpi=300)
plt.close()
