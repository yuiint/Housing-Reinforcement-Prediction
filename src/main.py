"""
Main execute function for housing reinforcement prediction.
"""
import os
import pandas as pd
from .utils import prepare_data, impute_data, predict_data_weigthed, hyper_turning, feature_importance_analysis


if __name__ == "__main__":
    ROOT = os.getcwd()
    DATAPATH = f"{ROOT}/data/train.csv"
    DATAPATH_preproc = f"{ROOT}/data/train_preproc.csv"
    
    scenarios = [
        ("Raw data", DATAPATH, False),
        ("Filfull missing value", DATAPATH, True),
        ("Preprocessed Data", DATAPATH_preproc, False),
        ("Preprocessed and imputed Data", DATAPATH_preproc, True),
    ]

    results_df = pd.DataFrame(columns=['Data', 'F1_Score', 'F1_Weighted'])

    for title, path, impute in scenarios:
        X, y = prepare_data(path)
        X = impute_data(X) if impute else X
        title, f1, f1_weighted = predict_data_weigthed(X, y, title)
        results_df.loc[len(results_df.index)] = [title, f1, f1_weighted]

    print("================== Outcome ==================")
    print(round(results_df, 4))

    print("\n================== Hyper turning ==================")
    X, y = prepare_data(DATAPATH)
    hyper_turning(X, y)

    print("\n================== Feature importance analysis ==================")
    X, y = prepare_data(DATAPATH)
    fs_best_f1_weighted = feature_importance_analysis(X, y)
    print(f"\nBest weighted f1 score after feature selection = {round(fs_best_f1_weighted, 4)}")