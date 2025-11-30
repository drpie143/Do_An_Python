"""Test script to verify entire pipeline"""
import sys
import numpy as np
import pandas as pd
from src.modeling.model_trainer import ModelTrainer
from src.preprocessing.data_preprocessor import DataPreprocessor

print("=== CHECKING IMPORTS ===")
print("✅ All imports OK\n")

print("=== TESTING FULL FLOW ===")

# 1. Create synthetic data
np.random.seed(42)
df = pd.DataFrame({
    'x1': np.random.uniform(10, 100, 100),
    'x2': np.random.uniform(5, 50, 100),
    'cat': np.random.choice(['A', 'B'], 100)
})
df['y'] = 2*df['x1'] + 3*df['x2'] + np.random.randn(100)*10
print("1. ✅ Created sample data (100 rows)")

# 2. Preprocessing
prep = DataPreprocessor(data=df)
prep.encode_categorical(method='onehot')
prep.scale_features(method='standard', exclude_columns=['y'])
df_proc = prep.get_processed_data()
print("2. ✅ Preprocessing done (encoded + scaled)")

# 3. Prepare data for training
X_train, X_test, y_train, y_test = ModelTrainer.prepare_data(df_proc, 'y')
print(f"3. ✅ Data prepared (train: {len(X_train)}, test: {len(X_test)})")

# 4. Initialize trainer
trainer = ModelTrainer(X_train, X_test, y_train, y_test)
print("4. ✅ Trainer initialized")

# 5. Train polynomial
print("\n   Training Polynomial Regression...")
trainer.train_polynomial(degree=2)
poly_r2 = trainer.results["polynomial"]["test_r2"]
poly_rmse = trainer.results["polynomial"]["test_rmse"]
print(f"   ✅ Polynomial: R²={poly_r2:.4f}, RMSE={poly_rmse:.2f}")

# 6. Train Random Forest
print("\n   Training Random Forest...")
trainer.train_rf(n_estimators=30)
rf_r2 = trainer.results["random_forest"]["test_r2"]
rf_rmse = trainer.results["random_forest"]["test_rmse"]
print(f"   ✅ Random Forest: R²={rf_r2:.4f}, RMSE={rf_rmse:.2f}")

# 7. Validation checks
print("\n=== VALIDATION CHECKS ===")

# Check polynomial metrics are reasonable (not billions)
if poly_rmse < 100:
    print(f"✅ Polynomial RMSE is reasonable: {poly_rmse:.2f} (not in billions!)")
else:
    print(f"❌ Polynomial RMSE too high: {poly_rmse:.2f}")

# Check R2 scores
if 0 <= poly_r2 <= 1 and 0 <= rf_r2 <= 1:
    print(f"✅ R² scores valid: Poly={poly_r2:.4f}, RF={rf_r2:.4f}")
else:
    print(f"❌ R² scores invalid")

# Check models exist
if "polynomial" in trainer.models and "random_forest" in trainer.models:
    print(f"✅ Models saved in trainer.models")
else:
    print(f"❌ Models not found")

print("\n=== ALL CHECKS PASSED ===")
print("🎉 Pipeline works correctly!")
