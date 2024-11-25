import pickle
import argparse
from pathlib import Path

import pandas as pd
from sklearn.base import BaseEstimator


def predict(model: BaseEstimator, features: pd.DataFrame) -> pd.Series:
    result = model.predict(features)
    result = pd.Series(result)
    return result


def load_model(path: Path) -> BaseEstimator:
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model


def load_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns='8')
    return df


def save_predictions(predictions: pd.Series):
    predictions.to_csv('predictions.csv')


def main():
    parser = argparse.ArgumentParser(description='Script for training decision tree regressor on csv. file data')
    parser.add_argument('--data_file_path', type=Path, help='Path to a data file')
    parser.add_argument('--model_file_path', type=Path, help='Path to a model file')
    args = parser.parse_args()

    print(args.model_file_path)
    model = load_model(args.model_file_path)
    features = load_features(args.data_file_path)

    result = predict(model, features)
    save_predictions(result)


if __name__ == '__main__':
    main()


