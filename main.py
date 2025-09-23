import logging
from pathlib import Path
from datetime import datetime

from src.utils.config import load_config, setup_logging
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.models.predictor import PowerPredictor
from src.models.evaluator import ModelEvaluator

def main():
    # Setup
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Brand Power Prediction Pipeline")
    
    try:
        # Load configuration
        config = load_config()
        
        # Create output directories
        Path("logs").mkdir(exist_ok=True)
        Path(config['data']['output_path']).mkdir(exist_ok=True)
        
        # Load data
        data_loader = DataLoader(config)
        df_train, df_test = data_loader.load_data()
        
        # Preprocess data
        preprocessor = DataPreprocessor(config)
        df_train, df_test = preprocessor.preprocess(df_train, df_test)
        
        # Train model
        predictor = PowerPredictor(config)
        val_score, y_val, val_pred, validation_data = predictor.train(df_train)
        
        # Evaluate model
        if config['evaluation']['calculate_pps']:
            evaluator = ModelEvaluator(config)
            metrics = evaluator.calculate_pps(y_val, val_pred, validation_data)
        
        # Make predictions
        df_test = predictor.predict(df_test)
        
        # Show sample predictions
        print("\nSample predictions:")
        print(df_test[['year', 'quarter', 'country', 'brand', 'power']].head(10))
        
        # Save results
        output_path = Path(config['data']['output_path']) / f"brand_power_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        submission = df_test[['year', 'quarter', 'country', 'brand', 'power']].copy()
        
        # Rename columns to match required format
        submission.columns = ['Year', 'Quarter', 'Country', 'Brand', 'Predicted Power']
        
        submission.to_csv(output_path, index=False)
        
        logger.info(f"Predictions saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()