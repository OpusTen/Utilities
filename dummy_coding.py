import pandas as pd
import os
import sklearn


def dummy_df(df,to_dummy_list):
    '''This function takes input dataframe raw, the list of columns to be dummy coded as inputs.
       Returns the dataframe with all the required columns dummy coded'''
    try:
        for item in to_dummy_list:
            dummies = pd.get_dummies(df[item],prefix=item,dummy_na=False)
            df = df.drop(item,1)
            df = pd.concat([df,dummies],axis=1)
    except Exception as e:
        print str(e)
    return df