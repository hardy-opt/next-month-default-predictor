import os
import sys
import pandas as pd
import joblib
from pathlib import Path

# Add src to path so we can import our modules
sys.path.append('src/')

from data_preprocessing import load_and_clean_data, preprocess_features, split_and_balance_data
from model_training import train_logistic_regression, train_random_forest, train_xgboost_with_tuning
from evaluation import evaluate_model, compare_models

def ensure_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        'data/raw', 'data/processed', 'models', 
        'results/figures', 'notebooks'
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def main():
    print("Starting Credit Card Default Prediction Pipeline...")
    
    # Ensure all directories exist
    ensure_directories()
    
    # Check if data file exists
    data_path = "data/raw/default_of_credit_card_clients.csv"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found!")
        print("Please move your default_of_credit_card_clients.csv file to data/raw/")
        return
    
    try:
        # Step 1: Load and preprocess data
        print("\n1. Loading and cleaning data...")
        df = load_and_clean_data(data_path)
        print(f"Dataset shape: {df.shape}")
        
        # Save cleaned data
        df.to_csv('data/processed/cleaned_data.csv', index=False)
        print("✓ Cleaned data saved to data/processed/")
        
        # Step 2: Prepare features
        print("\n2. Preprocessing features...")
        X, y, scaler = preprocess_features(df)
        
        # Save scaler
        joblib.dump(scaler, 'models/scaler.joblib')
        print("✓ Scaler saved to models/")
        
        # Step 3: Split and balance data
        print("\n3. Splitting and balancing data...")
        X_train, X_test, y_train, y_test = split_and_balance_data(X, y)
        joblib.dump(X_test, 'data/processed/X_test.pkl')
        joblib.dump(y_test, 'data/processed/y_test.pkl')

        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        
        # Step 4: Train models
        print("\n4. Training models...")
        
        print("   Training Logistic Regression...")
        logit_model = train_logistic_regression(X_train, y_train)
        joblib.dump(logit_model, 'models/logistic_regression.joblib')
        
        print("   Training Random Forest...")
        rf_model = train_random_forest(X_train, y_train)
        joblib.dump(rf_model, 'models/random_forest.joblib')
        
        print("   Training XGBoost with hyperparameter tuning...")
        xgb_model, best_params = train_xgboost_with_tuning(X_train, y_train)
        joblib.dump(xgb_model, 'models/xgboost.joblib')
        joblib.dump(best_params, 'models/xgb_best_params.joblib')
        
        
        print("✓ All models saved to models/")
        
        # Step 5: Evaluate models
        print("\n5. Evaluating models...")
        models = {
            "Logistic Regression": logit_model,
            "Random Forest": rf_model,
            "XGBoost": xgb_model
        }
        results_list = []
        # This will create plots and save them
        for name, model in models.items():
            print(f"\nEvaluating {name}...")
            result = evaluate_model(model, X_test, y_test, name)
            results_list.append(result)
        
        # Compare models
        compare_models(results_list, save_path='results/figures/model_comparison.png')
        print("\n" + "="*50)
        print("FINAL MODEL COMPARISON:")
        print("="*50)
        results = {r['model_name']: r['metrics']['accuracy'] for r in results_list}
        for model_name, accuracy in results.items():
            print(f"{model_name}: {accuracy:.4f}")
        
        # # # Save results
        # results_df = pd.DataFrame(list(results.items()), 
        #                         columns=['Model', 'Accuracy'])
        # results_df.to_csv('results/model_comparison.csv', index=False)
        results_df = pd.DataFrame([{'Model': r['model_name'],
                                    'Accuracy': r['metrics']['accuracy']
                                    } for r in results_list])
        results_df.to_csv('results/model_comparison.csv', index=False)
        print("\n✓ Results saved to results/model_comparison.csv")
        
        print("\n" + "="*50)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*50)

        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("Please check your data file and try again.")

if __name__ == "__main__":
    main()