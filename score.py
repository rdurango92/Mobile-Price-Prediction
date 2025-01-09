# score.py

import pandas as pd
import joblib
import logging

def main():
    """
    Loads and preprocesses data, loads model, generates 
    predictions, and logs to scores.log
    """

    # Load data to be scored
    X_val = pd.read_csv('data/scoring_data.csv')
    X_val = X_val.drop(['Unnamed: 0'], axis=1)

    # Preprocess data
    preprocessing_pipeline = joblib.load('models/preprocessing_pipeline.joblib')
    X_val = preprocessing_pipeline.transform(X_val)    

    # Load model
    model = joblib.load('models/model.joblib')
    
    # Generate scores
    scores = model.predict(X_val)
    
    # Log scores
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    f_handler = logging.FileHandler('scores.log')
    f_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)
    logger.info(f'Scores: {scores}')

if __name__ == '__main__':
    main()