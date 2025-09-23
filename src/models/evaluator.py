import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, config: dict):
        self.config = config
    
    def calculate_pps(self, y_true: np.ndarray, y_pred: np.ndarray, 
                     validation_data: pd.DataFrame) -> dict:
        """Calculate Power Prediction Score (PPS)."""
        logger.info("Calculating Power Prediction Score (PPS)...")
        
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # Calculate standard deviation of target variable
        sigma_y = np.std(y_true)
        
        # Calculate RMSE Skill Score
        rmse_skill = max(0, min(1, 1 - rmse/sigma_y))
        logger.info(f"RMSE: {rmse:.4f}")
        logger.info(f"Standard Deviation of Power: {sigma_y:.4f}")
        logger.info(f"RMSE Skill Score: {rmse_skill:.4f}")
        
        # Calculate Trend Hit Rate
        trend_hit_rate = self._calculate_trend_hit_rate(validation_data)
        
        # Calculate final PPS
        weights = self.config['evaluation']['pps_weights']
        pps = weights['rmse_skill'] * rmse_skill + weights['trend_hit_rate'] * trend_hit_rate
        logger.info(f"Power Prediction Score (PPS): {pps:.4f}")
        
        return {
            'pps': pps,
            'rmse': rmse,
            'rmse_skill': rmse_skill,
            'trend_hit_rate': trend_hit_rate,
            'sigma_y': sigma_y
        }
    
    def _calculate_trend_hit_rate(self, validation_data: pd.DataFrame) -> float:
        """Calculate trend hit rate."""
        trend_hits = 0
        total_cases = 0
        
        # Group by brand and country
        for (brand, country), group in validation_data.groupby(['brand', 'country']):
            # Sort by time
            group = group.sort_values('time_idx')
            
            # Need at least 2 points to calculate trend
            if len(group) < 2:
                continue
                
            # Calculate deltas for true and predicted values
            true_deltas = np.sign(np.diff(group['power_true']))
            pred_deltas = np.sign(np.diff(group['power_pred']))
            
            # Compare trends (direction of change)
            hits = (true_deltas == pred_deltas).sum()
            
            trend_hits += hits
            total_cases += len(true_deltas)
        
        # Calculate trend hit rate
        trend_hit_rate = trend_hits / total_cases if total_cases > 0 else 0
        logger.info(f"Trend Hits: {trend_hits} out of {total_cases}")
        logger.info(f"Trend Hit Rate: {trend_hit_rate:.4f}")
        
        return trend_hit_rate