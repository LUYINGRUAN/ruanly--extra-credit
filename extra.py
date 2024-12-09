import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
# 1. Load the data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 2. Feature Engineering: to change all into int type
train['trans_date'] = pd.to_datetime(train['trans_date']).astype('int64') // 10**9
train['hour'] = pd.to_datetime(train['trans_time']).dt.hour
train['minute'] = pd.to_datetime(train['trans_time']).dt.minute
train['second'] = pd.to_datetime(train['trans_time']).dt.second
train['age'] = pd.to_datetime('today').year - pd.to_datetime(train['dob']).dt.year
train.drop(columns=['trans_time', 'dob'], inplace=True, errors='ignore')
test['trans_date'] = pd.to_datetime(test['trans_date']).astype('int64') // 10**9
test['hour'] = pd.to_datetime(test['trans_time']).dt.hour
test['minute'] = pd.to_datetime(test['trans_time']).dt.minute
test['second'] = pd.to_datetime(test['trans_time']).dt.second
test['age'] = pd.to_datetime('today').year - pd.to_datetime(test['dob']).dt.year
test.drop(columns=['trans_time', 'dob'], inplace=True, errors='ignore')


# 3. Align features between train and test datasets
X_train = train.drop(columns=['is_fraud', 'id', 'trans_num'])  # Drop target, id, and trans_num
y_train = train['is_fraud']
X_test = test.drop(columns=['id', 'trans_num'])  # Test data (no 'is_fraud')

# Ensure the feature columns in both train and test are aligned in the same order
X_test = X_test[X_train.columns]

# 4. Encode categorical variables
object_cols = X_train.select_dtypes(include=['object']).columns  # Identify categorical columns
label_encoders = {}  # Dictionary to store label encoders

for col in object_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    X_test[col] = X_test[col].astype(str).apply(lambda x: x if x in le.classes_ else 'Unknown')
    le.classes_ = np.append(le.classes_, 'Unknown')  
    X_test[col] = le.transform(X_test[col])
    label_encoders[col] = le  

# 5. XGBoost Model Training with Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(X_train.shape[0])  
test_preds = np.zeros(X_test.shape[0])  

# Initialize the XGBoost model
model = xgb.XGBClassifier(
    objective='binary:logistic',  
    scale_pos_weight=2,  
    learning_rate=0.05,  
    n_estimators=1000,  
    max_depth=10,  
    min_child_weight=5,  
    subsample=0.8,  
    colsample_bytree=0.8,  
    gamma=0.1,  
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'  
)
# Train the model using StratifiedKFold cross-validation
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"Training fold {fold + 1}...")
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # Train the model
    model.fit(X_tr, y_tr, 
              eval_set=[(X_val, y_val)],   # Early stopping to avoid overfitting
              verbose=100)
    
    # Collect out-of-fold predictions
    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds += model.predict_proba(X_test)[:, 1] / skf.n_splits  # Averaging test set predictions across folds

# 6. Evaluate the performance using F1-Score
oof_preds_binary = (oof_preds > 0.5).astype(int)  # Convert probabilities to binary predictions
f1 = f1_score(y_train, oof_preds_binary)  # Calculate F1 score
print(f"Out-of-Fold F1 Score: {f1:.4f}")

# 7. Create the submission file
submission = pd.DataFrame({
    'id': test['id'],
    'is_fraud': (test_preds > 0.5).astype(int)  # Convert probabilities to binary predictions
})

# Save the predictions to a CSV file for submission
submission.to_csv('submission.csv', index=False)
print("Optimized predictions saved as 'submission.csv'")