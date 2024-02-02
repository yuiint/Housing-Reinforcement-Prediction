'''
General utilities for preprocessing and prediction.
'''
import os
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from tqdm import tqdm
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, make_scorer, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import KNNImputer


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


def predict_data_weigthed(X: pd.DataFrame, y: pd.DataFrame, title: str, show_performance: bool = True, save_prediction: bool = False) -> (str, float, float):
    '''
    Predict using XGBoost.
    '''  
    ROOT = os.getcwd()
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    class_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    f1_scorer = make_scorer(f1_score)   
    model = XGBClassifier(scale_pos_weight=class_weight, scorer=f1_scorer)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)

    f1 = f1_score(y_valid, y_pred)
    f1_weighted = f1_score(y_valid, y_pred, average='weighted')

    if show_performance:
        cm = confusion_matrix(y_valid, y_pred) 
        report = classification_report(y_valid, y_pred)
        outcome_message = (
            f"================== {title} ==================\n"
            f"F1 Score: {round(f1, 4)}\n"
            f"F1 weighted Score: {round(f1_weighted, 4)}\n"
            "\nConfusion Matrix:\n"
            f"{cm}\n"
            "\nReport:\n"
            f"{report}"
        )
        print(outcome_message)
    
    data_path = f"{ROOT}/data/X_test.csv"
    X_test = pd.read_csv(data_path)
    y_pred_test = model.predict(X_test[X_train.columns])

    if save_prediction:
        dic = {
            'id': X_test.index + 1,
            'label': y_pred_test
        }
        y_pred_test = pd.DataFrame(dic)
        y_pred_test.to_csv(f"{ROOT}/data/y_pred.csv", index=False)
        
    return title, f1, f1_weighted


def hyper_turning(X: pd.DataFrame, y: pd.DataFrame) -> None:
    '''
    Hyper-tuning by searching for the best max_depth, n_estimators, learning_rate, subsample.
    Best Parameters in prediction by raw data: 
    learning_rate=0.1, max_depth=6, n_estimators=300, subsample= 0.7
    Weighted f1 score = 0.7618
    '''
    class_weight = len(y[y == 0]) / len(y[y == 1])
    f1_scorer = make_scorer(f1_score)   
    model = XGBClassifier(scale_pos_weight=class_weight, scorer=f1_scorer)
    
    param_grid = {
        'max_depth': [3, 4, 5, 6, 7, 9],
        'n_estimators': [100, 200, 300, 400, 500],
        'learning_rate': [0.01, 0.1, 0.2, 0.5],
        'subsample': [0.5, 0.7, 1.0]
    }
    
    grid_search = GridSearchCV(model, param_grid, scoring=f1_scorer, cv=10)
    grid_search.fit(X, y)
    print("Best Parameters:", grid_search.best_params_)
    print(f"Best Weighted F1 Score: {round(grid_search.best_score_, 4)}")


def feature_importance_analysis(X: pd.DataFrame, y: pd.DataFrame) -> float:
    '''
    Analyze the F1 score's performance as the number of features increases.
    '''
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    class_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    f1_scorer = make_scorer(f1_score)   
    model = XGBClassifier(scale_pos_weight=class_weight, scorer=f1_scorer)
    model.fit(X_train, y_train)

    xgb.plot_importance(model)
    plt.show()

    booster = model.get_booster()
    feature_importance = booster.get_score(importance_type='weight')
    sorted_feature_importance = {k: v for k, v in sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)}
    feature_names = list(sorted_feature_importance.keys())

    f1_score_list = []
    for i, _ in tqdm(enumerate(feature_names, start=1)):
        feature_selected = X[feature_names[:i]]
        _, _, f1_weighted = predict_data_weigthed(feature_selected, y, f'{i} features', False)
        f1_score_list.append(f1_weighted)

    x_ax = list(range(1, len(feature_names)+1))
    y_ax = f1_score_list

    plt.plot(x_ax, y_ax)
    plt.title("F1 score's performance as the number of features increases")
    plt.ylabel("Weighted f1 score ")
    plt.xlabel("Number of features")
    plt.show()

    return max(f1_score_list)