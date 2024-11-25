import pickle
import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator


def data_preprocessing(path: Path) -> tuple[pd.DataFrame, Optional[pd.Series]]:
    df = pd.read_csv(path)
    df = df.drop(columns='8')
    features = df.drop(columns='target')
    target = df['target']
    return features, target


def get_trained_regressor(features: pd.DataFrame, target: pd.Series) -> BaseEstimator:
    tree_regressor = DecisionTreeRegressor(random_state=42)
    tree_regressor.fit(features, target)
    return tree_regressor


def save_model(model: BaseEstimator):
    with open(f'./model.pkl', 'wb') as f:
        pickle.dump(model, f)


def main():
    parser = argparse.ArgumentParser(description='Script for training decision tree regressor on csv. file data')
    parser.add_argument('--file_path', type=Path, help='Path to a data file')
    args = parser.parse_args()

    features, target = data_preprocessing(args.file_path)
    model = get_trained_regressor(features, target)
    save_model(model)


if __name__ == '__main__':
    main()

