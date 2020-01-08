# data source: https://www.kaggle.com/uciml/student-alcohol-consumption

import pandas as pd

def get_alcohol_data(path = 'input/student-mat.csv'):
    df = pd.read_csv(path) # TODO: get portoguese students (not only math)
    above_average_grade = df['G3'] > df['G3'].mean()
    above_average_grade = above_average_grade.astype(int)

    categorical_variables = list(df.columns[df.dtypes == 'object'])

    df.insert(0, 'above_average_grade', above_average_grade)
    df = df.drop(columns = ['G1', 'G2', 'G3'])
    df_with_dummies = pd.get_dummies(df, columns=categorical_variables, drop_first=True)

    print('alcohol data import succesful')
    return df_with_dummies
