# Data source: https://www.kaggle.com/mohansacharya/graduate-admissions

import pandas as pd

def get_admission_data(path = 'input/Admission_Predict_Ver1.1.csv'):
    df = pd.read_csv(path)
    above_median_chance = df['Chance of Admit '] > df['Chance of Admit '].median()
    df.insert(0, 'above_median_chance', above_median_chance)
    df = df.drop(columns=['Serial No.', 'Chance of Admit '])
    df.above_median_chance = df.above_median_chance.astype(int)

    print('admission data import succesful')
    return df
