import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Tuple
import logging

from src.utils.outliers import calculate_iqr_bounds, cap_outliers

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, config: dict):
        self.config = config
        self.label_encoders = {}
        
    def preprocess(self, df_train: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Main preprocessing pipeline."""
        logger.info("Starting data preprocessing...")
        
        # Standardize column names
        df_test = self._standardize_columns(df_test)
        
        # Add time features
        df_train = self._add_time_features(df_train)
        df_test = self._add_time_features(df_test)
        
        # Create combined reference
        df_train, df_test = self._create_combined_reference(df_train, df_test)
        
        # Estimate brand metrics for test data
        df_test = self._estimate_brand_metrics(df_train, df_test)
        
        # Apply outlier capping if enabled
        if self.config['preprocessing']['apply_outlier_capping']:
            df_train = self._apply_outlier_capping(df_train)
        
        # Encode categorical features
        df_train, df_test = self._encode_categorical(df_train, df_test)
        
        logger.info("Data preprocessing completed")
        return df_train, df_test
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names."""
        if 'Year' in df.columns:
            df = df.rename(columns={
                'Year': 'year', 
                'Quarter': 'quarter', 
                'Country': 'country', 
                'Brand': 'brand', 
                'Predicted Power': 'predicted_power'
            })
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        df = df.copy()
        df['quarter_num'] = df['quarter'].str.replace('Qtr', '').astype(int)
        df['time_idx'] = df['year'] * 10 + df['quarter_num']
        return df
    
    def _create_combined_reference(self, df_train: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create combined dataset for reference."""
        df_test['is_test'] = True
        df_train['is_test'] = False
        combined = pd.concat([df_train, df_test], ignore_index=True)
        combined = combined.sort_values(['brand', 'country', 'year', 'quarter_num'])
        
        logger.info(f"Training data: {len(df_train)} rows")
        logger.info(f"Test data: {len(df_test)} rows")
        
        return df_train, df_test
    
    def _estimate_brand_metrics(self, df_train: pd.DataFrame, df_test: pd.DataFrame) -> pd.DataFrame:
        """Estimate brand metrics for test data."""
        logger.info("Estimating brand metrics for test data...")
        
        df_test = df_test.copy()
        
        # Calculate averages by brand and country
        brand_avgs = df_train.groupby('brand').agg({
            'meaning': 'mean',
            'difference': 'mean',
            'salience': 'mean',
            'premium': 'mean'
        })
        
        country_avgs = df_train.groupby('country').agg({
            'meaning': 'mean',
            'difference': 'mean',
            'salience': 'mean',
            'premium': 'mean'
        })
        
        # Global averages (fallback)
        global_avgs = {
            'meaning': df_train['meaning'].mean(),
            'difference': df_train['difference'].mean(),
            'salience': df_train['salience'].mean(),
            'premium': df_train['premium'].mean()
        }
        
        # Fill in metrics for each test row
        metrics = ['meaning', 'difference', 'salience', 'premium']
        for idx, row in df_test.iterrows():
            brand, country = row['brand'], row['country']
            
            # Try brand average first
            if brand in brand_avgs.index:
                for metric in metrics:
                    df_test.loc[idx, metric] = brand_avgs.loc[brand, metric]
            
            # Then try country average
            elif country in country_avgs.index:
                for metric in metrics:
                    df_test.loc[idx, metric] = country_avgs.loc[country, metric]
            
            # Fallback to global average
            else:
                for metric in metrics:
                    df_test.loc[idx, metric] = global_avgs[metric]
        
        return df_test
    
    def _apply_outlier_capping(self, df_train: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply outlier capping using IQR method."""
        logger.info("Applying outlier capping...")
        
        # Only use train data with non-null 'power'
        train_data = df_train[df_train['power'].notnull()].copy()
        
        # Apply outlier capping to ALL numeric columns including target variable
        numeric_cols = ['meaning', 'difference', 'salience', 'premium', 'power']
        iqr_bounds = calculate_iqr_bounds(train_data, numeric_cols)
        train_data_capped = cap_outliers(train_data, iqr_bounds)
        
        # Update the original train dataframe
        df_train_result = df_train.copy()
        df_train_result.loc[df_train['power'].notnull()] = train_data_capped
        
        return df_train_result
    
    def _encode_categorical(self, df_train: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Encode categorical features."""
        logger.info("Encoding categorical features...")
        
        df_train = df_train.copy()
        df_test = df_test.copy()
        
        for col in ['country', 'brand']:
            le = LabelEncoder()
            df_train[f'{col}_encoded'] = le.fit_transform(df_train[col])
            df_test[f'{col}_encoded'] = le.transform(df_test[col])
            self.label_encoders[col] = le
        
        return df_train, df_test