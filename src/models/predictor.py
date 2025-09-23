import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from typing import Tuple
import logging
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

class PowerPredictor:
    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.feature_columns = None
        
    def train(self, df_train: pd.DataFrame) -> Tuple[float, np.ndarray, np.ndarray, pd.DataFrame]:
        """Train the model and return validation results."""
        logger.info("Preparing data for prediction model...")
        
        # Only use train data with non-null 'power'
        train_data = df_train[df_train['power'].notnull()].copy()
        
        # Select features and target
        self.feature_columns = ['year', 'quarter_num', 'country_encoded', 'brand_encoded', 
                               'meaning', 'difference', 'salience', 'premium']
        target = 'power'
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            train_data[self.feature_columns], 
            train_data[target], 
            test_size=self.config['preprocessing']['validation_split'], 
            random_state=self.config['preprocessing']['random_state']
        )
        
        # Train model
        self.model = Ridge(alpha=self.config['model']['ridge']['alpha'])
        self.model.fit(X_train, y_train)
        
        # Evaluate on validation set
        val_pred = self.model.predict(X_val)
        val_score = r2_score(y_val, val_pred)
        logger.info(f"Model R² score on validation data: {val_score:.4f}")
        
        # Prepare validation data for trend analysis
        val_data = train_data.loc[X_val.index].copy()
        val_data['power_true'] = y_val
        val_data['power_pred'] = val_pred
        
        return val_score, y_val, val_pred, val_data
    
    def predict(self, df_test: pd.DataFrame) -> pd.DataFrame:
        """Make predictions on test data."""
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        logger.info("Making predictions on test set...")
        
        df_test = df_test.copy()
        df_test['power'] = self.model.predict(df_test[self.feature_columns])
        
        # Apply minimum value constraint (power cannot be negative)
        df_test['power'] = df_test['power'].clip(lower=self.config['preprocessing']['power_min_value'])
        
        return df_test
    
    def save_model(self, filepath: str):
        """Save trained model."""
        if self.model is None:
            raise ValueError("No model to save")
        
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'config': self.config
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")