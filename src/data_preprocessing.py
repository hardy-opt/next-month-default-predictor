import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def load_and_clean_data(filepath):
    """Load and clean the credit card dataset"""
    df = pd.read_csv(filepath)
    df.drop(["ID"], axis=1, inplace=True)
    
    # Clean categorical variables
    df['EDUCATION'].replace({0:1,1:1,2:2,3:3,4:4,5:1,6:1}, inplace=True)
    df['MARRIAGE'].replace({0:1,1:1,2:2,3:3}, inplace=True)
    
    return df

def preprocess_features(df):
    """Separate features and target, scale features"""
    X = df.drop(['default.payment.next.month'], axis=1)
    y = df['default.payment.next.month']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

def split_and_balance_data(X, y, test_size=0.20, random_state=42):
    """Split data and apply SMOTE"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    smote = SMOTE()
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    return X_train_balanced, X_test, y_train_balanced, y_test