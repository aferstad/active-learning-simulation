import pandas as pd


def get_heart_data(path='input/heart/heart.csv'):
    heart = pd.read_csv(path)

    # move label column to first position:
    target = heart.target
    heart = heart.drop(columns=['target'])
    heart.insert(0, 'has_disease', target)

    # rest index to make sure index is unique:
    heart = heart.reset_index().drop(columns=['index'])

    heart_with_dummies = pd.get_dummies(heart, columns=['cp', 'thal', 'slope'], drop_first=True)

    print('heart data import succesful')
    return heart_with_dummies
