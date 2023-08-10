import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import mlflow
import mlflow.sklearn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    logger.info(f'Loaded {len(df)} rows from {path}')
    return df


def preprocess(df: pd.DataFrame, target_col: str):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)


def train(model, X_train, y_train, X_test, y_test, run_name: str):
    with mlflow.start_run(run_name=run_name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        report = classification_report(y_test, preds, output_dict=True)
        mlflow.log_metrics({'accuracy': report['accuracy']})
        mlflow.sklearn.log_model(model, 'model')
        logger.info(classification_report(y_test, preds))
    return model


if __name__ == '__main__':
    from sklearn.ensemble import GradientBoostingClassifier
    df = load_data('data/dataset.csv')
    X_train, X_test, y_train, y_test = preprocess(df, target_col='label')
    model = GradientBoostingClassifier(n_estimators=200, max_depth=5)
    train(model, X_train, y_train, X_test, y_test, run_name='gbm_baseline')
