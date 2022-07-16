from sklearn.metrics import mean_absolute_error
from data_preprocessing import get_train_test

import logging
import pickle
import pandas as pd

def load_models():
    #load models
    DecisionTreeRegressorFilename = "DecisionTreeRegressor.sav"
    model1 = pickle.load(open("./models/" + DecisionTreeRegressorFilename, 'rb'))
    logging.info(f'Model {DecisionTreeRegressorFilename} is loaded from disk successfully Using Pickle')

    KNeighborsRegressorFilename = "KNeighborsRegressor.sav"
    model2 = pickle.load(open("./models/" + KNeighborsRegressorFilename,'rb'))
    logging.info(f'Model {KNeighborsRegressorFilename} is loaded from disk successfully Using Pickle')

    return model1, model2

def validate_models(X_final_test: pd.DataFrame, y_final_test: pd.DataFrame) -> dict:
    model1, model2 = load_models()
    run_results = {}
    preds1 = model1.predict(X_final_test)
    preds2 = model2.predict(X_final_test).reshape(len(X_final_test))
    preds = (preds1+preds2)/2
    model_mae = mean_absolute_error(y_final_test,preds)
    run_results['final_model'] = model_mae

    return run_results


def predict(X: pd.Dataframe) -> pd.DataFrame:
    model1, model2 = load_models()
    preds1 = model1.predict(X_final_test)
    preds2 = model2.predict(X_final_test).reshape(len(X_final_test))
    preds = (preds1+preds2)/2

    return pd.DataFrame(preds)


if __name__ == '__main__':
    X, X_final_test, y, y_final_test = get_train_test()


