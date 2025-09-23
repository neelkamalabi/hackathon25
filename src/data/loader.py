import pandas as pd
from pathlib import Path
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, config: dict):
        self.config = config
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load training and test data."""
        try:
            train_path = self.config['data']['train_path']
            test_path = self.config['data']['test_path']
            
            logger.info(f"Loading training data from {train_path}")
            df_train = pd.read_csv(train_path)
            
            logger.info(f"Loading test data from {test_path}")
            df_test = pd.read_csv(test_path)
            
            logger.info(f"Loaded {len(df_train)} training samples and {len(df_test)} test samples")
            return df_train, df_test
            
        except FileNotFoundError as e:
            logger.error(f"Data file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise