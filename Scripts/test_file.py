import pandas as pd
from Imputer import imputer
from BestRegressor import get_best_reg

data = pd.read_csv('../Data/HousePrices/train.csv')

data = imputer(data)

data.drop("Id", axis=1, inplace=True)

data = pd.get_dummies(data)

target = 'SalePrice'

features = [x for x in data.columns if x != target]

x = data.loc[:, features]
y = data.loc[:, target]

best_regressor, training_log = get_best_reg(x, y)

