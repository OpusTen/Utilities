import pandas as pd
import numpy as np
import sys


def imputer(data):
    """
    Function takes raw dataframe and imputes both numerical and categorical values, also removes columns with 90 percent missing data

    For Numerical types( int, float ):
        * checks the distribution of the data
        * if skew is within limits, then mean value is used as imputer
        * otherwise, median is used

    For Categorical types ( str )
        * most frequent value is used

    :param data: dataframe with raw values
    :return : imputed data
    """
    try:
        if data.empty:
            sys.exit('Empty Dataframe received...Terminating Execution')

        """
        Dealing with Missing Values
        """
        treshold = 0.80

        missing_value_df = pd.DataFrame(data.isnull().sum().sort_values(ascending=False) / len(data))
        missing_value_df.reset_index(inplace=True)
        missing_value_df.columns = ['Feature', 'Missing Value Ratio']

        features_to_be_dropped = list(
            missing_value_df[missing_value_df['Missing Value Ratio'] >= treshold].loc[:, 'Feature'].values)

        if len(features_to_be_dropped) > 0:

            print("Following features having more than " + str(treshold) + " missing data : {}".format(
                features_to_be_dropped))
            print("Dropping features...")

            data.drop(features_to_be_dropped, axis=1, inplace=True)
        else:

            print("No Features dropped...")

        """
        Segregating Numeric & Categorical Columns 
        """

        category_cols = []
        numerical_cols = []

        for col_name in list(data.columns):
            if (data[col_name].dtype == np.float64 or data[col_name].dtype == np.int64):
                numerical_cols.append(col_name)
            elif data[col_name].dtype == object:
                category_cols.append(col_name)

        """
        Numerical Data Imputation
        """

        for feature in numerical_cols:

            num_missing_values = data[feature].isnull().sum()

            if num_missing_values > 0:

                print("Feature Name : {}".format(feature))
                print("\t Number of Missing Fields : {}".format(data[feature].isnull().sum()))

                skew = data[feature].skew()
                print("\t Skew of Distribution : {}".format(skew))

                if (skew <= 0.5) and (skew > -0.5):
                    fill_with = data[feature].mean()
                    print("\t Imputing with Mean : {}".format(fill_with))

                else:
                    fill_with = data[feature].median()
                    print("\t Imputing with Median : {}".format(fill_with))

                data[feature].fillna(fill_with, inplace=True)

        """
        Categorical Data Imputation
        """

        for feature in category_cols:
            num_missing_values = data[feature].isnull().sum()

            if num_missing_values > 0:
                print("Feature Name : {}".format(feature))
                print("\t Number of Missing Fields : {}".format(data[feature].isnull().sum()))

                fill_with = data[feature].value_counts().index[0]
                print("\t Imputing with Mode : {}".format(fill_with))

                data[feature].fillna(fill_with, inplace=True)

    except Exception as e:
        print(str(e))

    return data
