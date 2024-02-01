"""
Main execute function for housing reinforcement prediction.
"""
import os
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from tqdm import tqdm
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, make_scorer, confusion_matrix, classification_report
from sklearn.impute import KNNImputer
from sklearn.model_selection import GridSearchCV


def prepare_data(data_path: str) -> (pd.DataFrame, pd.DataFrame):
    '''
    Prepare data for prediction.
    '''
    data = pd.read_csv(data_path)
    X = data.iloc[:, :-2]
    y = data.iloc[:, -1]
    return X, y


def impute_data(X: pd.DataFrame, n_neighbors=7) -> pd.DataFrame:
    '''
    Fill missing values with KNNImputer.
    '''
    imputer = KNNImputer(n_neighbors=n_neighbors)
    X_imputed = imputer.fit_transform(X)
    return pd.DataFrame(X_imputed, columns=X.columns)


def predict_data_weigthed(X: pd.DataFrame, y: pd.DataFrame, title: str) -> (str, float, float):
    '''
    Predict using XGBoost.
    '''  
    print(f"================== {title} ==================")

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    class_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    f1_scorer = make_scorer(f1_score)   
    model = XGBClassifier(scale_pos_weight=class_weight, scorer=f1_scorer)

    # Train model.
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)

    # Model performance.
    f1 = f1_score(y_valid, y_pred)
    print("F1 Score: {:.4f}".format(f1))   
    f1_weighted = f1_score(y_valid, y_pred, average='weighted')
    print("F1 weighted Score: {:.4f}".format(f1_weighted))    

    # Confusion Matrix
    cm = confusion_matrix(y_valid, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Report
    print("\nReport:")
    print(classification_report(y_valid,y_pred))
    
    # Predict
    data_path = f"{ROOT}/data/X_test.csv"
    X_test = pd.read_csv(data_path)

    # Save outcome
    y_pred_test = model.predict(X_test[X_train.columns])
    dic = {
        'id': X_test.index + 1,
        'label': y_pred_test
    }
    y_pred_test = pd.DataFrame(dic)
    y_pred_test.to_csv(f"{ROOT}/data/y_pred.csv", index=False)
    
    # Result
    print('1:', len(y_pred_test[y_pred_test['label']==1]))
    print('0:', len(y_pred_test[y_pred_test['label']==0]))

    return title, f1, f1_weighted


def hyper_turning(X: pd.DataFrame, y: pd.DataFrame) -> None:
    '''
    Hyper-tuning by searching for the best max_depth, n_estimators, learning_rate, subsample.
    Best Parameters in prediction by raw data: 
    learning_rate=0.1, max_depth=6, n_estimators=300, subsample= 0.7
    Weighted f1 score = 0.7618
    '''
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    class_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    f1_scorer = make_scorer(f1_score)   
    model = XGBClassifier(scale_pos_weight=class_weight, scorer=f1_scorer)

    param_grid = {
        'max_depth': [3, 4, 5, 6, 7, 9],
        'n_estimators': [100, 200, 300, 400, 500],
        'learning_rate': [0.01, 0.1, 0.2, 0.5],
        'subsample': [0.5, 0.7, 1.0]
    }

    # Grid Search
    grid_search = GridSearchCV(model, param_grid, scoring=f1_weighted_scorer, cv=10)
    grid_search.fit(X, y)
    print("Best Parameters:", grid_search.best_params_)
    print("Best Weighted F1 Score:", grid_search.best_score_)


def feature_importance_analysis(X: pd.DataFrame, y: pd.DataFrame) -> float:
    '''
    Analyze the F1 score's performance as the number of features increases.
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    class_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    f1_scorer = make_scorer(f1_score)   
    model = XGBClassifier(scale_pos_weight=class_weight, scorer=f1_scorer)
    # model = XGBClassifier(objective='binary:logistic', scale_pos_weight=class_weight, scorer=f1_weighted_scorer)
    model.fit(X_train, y_train)

    # Visualize feature importance.
    xgb.plot_importance(model)
    plt.show()

    # Get feature importance list.
    booster = model.get_booster()
    feature_importance = booster.get_score(importance_type='weight')
    sorted_feature_importance = {k: v for k, v in sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)}
    feature_names = list(sorted_feature_importance.keys())

    f1_score_list = []
    for i in tqdm(range(1, len(feature_names))):
        X_selected = X[feature_names[:i]]
        title, f1, f1_weighted = predict_data_weigthed(X_selected, y, f'{i} features')
        f1_score_list.append(f1_weighted)

    x_ax = list(range(1, len(feature_names)))
    y_ax = f1_score_list

    # Check out the F1 score's performance as the number of features increases.
    plt.plot(x_ax, y_ax)
    plt.ylabel("Weighted f1 score ")
    plt.xlabel("Number of features")
    plt.show()

    return max(f1_score_list)


if __name__ == "__main__":
    ROOT = os.path.dirname(os.getcwd())
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
    print(results_df)

    print("\n================== Hyper turning ==================")
    X, y = prepare_data(DATAPATH)
    hyper_turning(X, y)

    print("\n================== Feature importance analysis ==================")
    X, y = prepare_data(DATAPATH)
    fs_best_f1_weighted = feature_importance_analysis(X, y)
    print(f"Best weighted f1 score after feature selection = {round(fs_best_f1_weighted, 4)}")