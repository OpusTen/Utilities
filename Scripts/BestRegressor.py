from sklearn.linear_model import LinearRegression,RidgeCV, LassoCV, ElasticNetCV
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

import pandas as pd
from matplotlib import pyplot as plt

def get_best_reg(x, y, val_size=0.20):
    """

    :param x: features dataframe
    :param y: target dataframe/series
    :param val_size: size of the validation set
    :return:
    """

    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=val_size)

    regressors = [
        LinearRegression(),
        RidgeCV(),
        LassoCV(),
        ElasticNetCV(),
        SVR(),
        KNeighborsRegressor(),
        MLPRegressor(),
        DecisionTreeRegressor(),
        RandomForestRegressor(),
        AdaBoostRegressor(),
        GradientBoostingRegressor(),
        XGBRegressor()]

    # Logging for Visual Comparison
    log_cols = ["Regressor", "Training Error", "Validation Error"]
    log = pd.DataFrame(columns=log_cols)

    for reg in regressors:
        print("=" * 30)

        reg_name = reg.__class__.__name__
        print(reg_name)

        reg.fit(x_train, y_train)

        # Training Accuracy
        y_train_pred = reg.predict(x_train)
        train_err = mean_absolute_error(y_train, y_train_pred)

        # Validation Accuracy
        y_valid_pred = reg.predict(x_valid)
        valid_err = mean_absolute_error(y_valid, y_valid_pred)

        print("Validation Error: {}".format(valid_err))

        log_entry = pd.DataFrame([[reg_name, train_err, valid_err]], columns=log_cols)
        log = log.append(log_entry, ignore_index=True)

    log.sort_values('Validation Error', ascending=True, inplace=True)

    #log.plot.barh(x='Regressor', y=['Training Error', 'Validation Error'])
    #plt.show()

    best_model = log.loc[log['Validation Error'] == log['Validation Error'].min(), 'Regressor'].iloc[0]

    print("Best Regressor : {}".format(best_model))

    return best_model, log