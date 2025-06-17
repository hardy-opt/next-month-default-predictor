from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV

def train_logistic_regression(X_train, y_train):
    """Train logistic regression model"""
    logit = LogisticRegression()
    logit.fit(X_train, y_train)
    return logit

def train_random_forest(X_train, y_train):
    """Train random forest model"""
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    return rf

def train_xgboost_with_tuning(X_train, y_train):
    """Train XGBoost with hyperparameter tuning"""
    xgb_clf = xgb.XGBClassifier()
    
    params = {
        "learning_rate": [0.001, 0.005, 0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
        # ... rest of params
    }
    
    random_search = RandomizedSearchCV(
        xgb_clf, param_distributions=params, n_iter=5, 
        scoring='roc_auc', n_jobs=-1, cv=5, verbose=3
    )
    
    random_search.fit(X_train, y_train)
    
    # Return best model
    best_xgb = xgb.XGBClassifier(**random_search.best_params_)
    best_xgb.fit(X_train, y_train)
    
    return best_xgb, random_search.best_params_