import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load your filtered dataset
df = pd.read_csv("../data/filtered_2015_onwards.csv", low_memory=False)

# ----- 1. Select useful columns -----
selected_columns = [
    'loan_amnt', 'term', 'int_rate', 'installment', 'grade', 'sub_grade',
    'emp_length', 'home_ownership', 'annual_inc', 'purpose', 'addr_state',
    'dti', 'delinq_2yrs', 'fico_range_high', 'fico_range_low',
    'open_acc', 'pub_rec', 'revol_util', 'total_acc', 'issue_d',
    'loan_status'
]

df = df[selected_columns]

# ----- 2. Drop rows with null target -----
df = df[df['loan_status'].notna()]

# ----- 3. Simplify target -----
# Define 'Charged Off' or similar as default, else 'Fully Paid' as non-default
df['loan_status'] = df['loan_status'].apply(lambda x: 1 if x in ['Charged Off', 'Default'] else 0)

# ----- 4. Define features -----
categorical_features = ['term', 'grade', 'sub_grade', 'emp_length', 'home_ownership', 'purpose', 'addr_state']
numeric_features = ['loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti',
                    'delinq_2yrs', 'fico_range_high', 'fico_range_low', 'open_acc',
                    'pub_rec', 'revol_util', 'total_acc']

# ----- 5. Preprocessing pipeline -----
numeric_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_pipeline, numeric_features),
    ('cat', categorical_pipeline, categorical_features)
])

# ----- 6. Split and transform -----
X = df[numeric_features + categorical_features]
y = df['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

joblib.dump(preprocessor, 'preprocessor.pkl')

print("✅ Preprocessing complete. Shapes:")
print("X_train:", X_train_processed.shape)
print("X_test:", X_test_processed.shape)

