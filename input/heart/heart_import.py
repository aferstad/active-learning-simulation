# https://archive.ics.uci.edu/ml/datasets/Heart+Disease
"""
    age: age in years
    sex: sex (1 = male; 0 = female)
    cp: chest pain type
        -- Value 1: typical angina
        -- Value 2: atypical angina
        -- Value 3: non-anginal pain
        -- Value 4: asymptomatic
    trestbps: resting blood pressure (in mm Hg on admission to the
        hospital)
    chol: serum cholestoral in mg/dl
    fbs: (fasting blood sugar > 120 mg/dl)  (1 = true; 0 = false)
    restecg: resting electrocardiographic results
        -- Value 0: normal
        -- Value 1: having ST-T wave abnormality (T wave inversions and/or ST
                    elevation or depression of > 0.05 mV)
        -- Value 2: showing probable or definite left ventricular hypertrophy
                    by Estes' criteria
    thalach: maximum heart rate achieved
    exang: exercise induced angina (1 = yes; 0 = no)
    oldpeak = ST depression induced by exercise relative to rest
    slope: the slope of the peak exercise ST segment
        -- Value 1: upsloping
        -- Value 2: flat
        -- Value 3: downsloping
    ca: number of major vessels (0-3) colored by flourosopy
    thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
    disease_presence:  integer valued from 0 (no presence) to 4
"""

import pandas as pd

def get_heart_data(path='input/heart/processed.cleveland.data'):
    df = pd.read_csv(path, header = None)
    df.columns = ['age',
        'sex',
        'cp',
        'trestbps',
        'chol',
        'fbs',
        'restecg',
        'thalach',
        'exang',
        'oldpeak',
        'slope',
        'ca',
        'thal',
        'disease_present']

     # move label column to first position:
    disease_present = df.disease_present
    df = df.drop(columns=['disease_present'])
    df.insert(0, 'disease_present', disease_present)
    df.loc[df.disease_present > 0, 'disease_present'] = 1  # convert presence to 0 or 1

    # rest index to make sure index is unique:
    df = df.reset_index().drop(columns=['index'])

    df_with_dummies = pd.get_dummies(df, columns=['cp','restecg','slope', 'ca', 'thal'], drop_first=True)

    print('heart data import succesful')
    return df_with_dummies
