# xgboost_house_price_prediction.py

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ------------------------------
# Step 1: Load and Clean Training Data
# ------------------------------
train = pd.read_csv("train.csv")

cols_to_drop = ['PoolQC', 'MiscFeature', 'Alley', 'Fence']
cat_cols = ['FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
            'BsmtExposure', 'BsmtFinType2', 'BsmtQual', 'BsmtCond', 'BsmtFinType1', 'MasVnrType']

train.drop(cols_to_drop, axis=1, inplace=True)
train[cat_cols] = train[cat_cols].fillna("None")
train['MasVnrArea'].fillna(0, inplace=True)
train['GarageYrBlt'].fillna(0, inplace=True)
train['LotFrontage'].fillna(train['LotFrontage'].median(), inplace=True)
train['Electrical'].fillna(train['Electrical'].mode()[0], inplace=True)

# ------------------------------
# Step 2: Feature Engineering and Encoding
# ------------------------------
X = train.drop(['Id', 'SalePrice'], axis=1)
y = np.log1p(train['SalePrice'])  # log-transform target

X = pd.get_dummies(X)

# ------------------------------
# Step 3: Train-Validation Split
# ------------------------------
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------
# Step 4: Train XGBoost Model with GridSearch
# ------------------------------
param_grid = {
    'n_estimators': [200],
    'max_depth': [3],
    'learning_rate': [0.1],
    'subsample': [1],
    'colsample_bytree': [0.8]
}

grid_search = GridSearchCV(
    estimator=XGBRegressor(random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring='neg_mean_absolute_error',
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# ------------------------------
# Step 5: Evaluate Model
# ------------------------------
preds_log = best_model.predict(X_valid)
preds = np.expm1(preds_log)
y_valid_exp = np.expm1(y_valid)

mae = mean_absolute_error(y_valid_exp, preds)
print("MAE after log transformation:", mae)

# Plot Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_valid_exp, preds, alpha=0.3)
plt.plot([min(y_valid_exp), max(y_valid_exp)], [min(y_valid_exp), max(y_valid_exp)], color='red', linestyle='--')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()

# ------------------------------
# Step 6: Load and Preprocess Test Data
# ------------------------------
test = pd.read_csv("test.csv")
test.drop(cols_to_drop, axis=1, inplace=True)
test[cat_cols] = test[cat_cols].fillna("None")
test['MasVnrArea'].fillna(0, inplace=True)
test['GarageYrBlt'].fillna(0, inplace=True)
test['LotFrontage'].fillna(test['LotFrontage'].median(), inplace=True)
test['Electrical'].fillna(test['Electrical'].mode()[0], inplace=True)

X_test = pd.get_dummies(test)
X_test = X_test.reindex(columns=X.columns, fill_value=0)

# ------------------------------
# Step 7: Predict on Test Data and Save Submission
# ------------------------------
test_preds_log = best_model.predict(X_test)
test_preds = np.expm1(test_preds_log)

submission = pd.DataFrame({
    'Id': test['Id'],
    'SalePrice': test_preds
})

submission.to_csv("submission.csv", index=False)
print("submission.csv created successfully.")
