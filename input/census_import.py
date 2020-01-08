import pandas as pd
#import seaborn as sns

def get_census_data(path_train, path_test):
    # import:
    columns = [
        "age", "workClass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country",
        "income"
    ]
    census_train = pd.read_csv(path_train,
                               names=columns,
                               sep=' *, *',
                               na_values='?')
    census_test = pd.read_csv(path_test,
                              names=columns,
                              sep=' *, *',
                              skiprows=1,
                              na_values='?')

    census = pd.concat([census_train, census_test], axis=0)

    # clean:
    census['income'] = census['income'].map(lambda x: x.replace('.', ''))

    census['income_over_50K'] = None
    census.loc[census.income == '<=50K', 'income_over_50K'] = 0
    census.loc[census.income == '>50K', 'income_over_50K'] = 1

    income_over_50K = census.income_over_50K
    census = census.drop(columns=['income_over_50K', 'income'])
    census.insert(0, 'income_over_50K', income_over_50K)

    census = census.drop(columns = ['fnlwgt']) # does not have anything to do with data

    # Making native-country to country_USA feature instead:
    census['country_USA'] = 0
    census.loc[census['native-country'] == 'United-States', 'country_USA'] = 1
    census = census.drop(columns=['native-country'])

    census = census.reset_index().drop(columns=['index'])

    census_with_dummies = pd.get_dummies(data= census,columns=census.select_dtypes(include=['object']).columns,drop_first=True)

    return census, census_with_dummies

def explore_data(data, label):
    num_attributes = data.select_dtypes(include=['int'])
    print('Numerical features: ' + str(num_attributes.columns))

    num_attributes.hist(figsize=(10,10))

    print(data.describe())

    cat_attributes = data.select_dtypes(include=['object'])
    print('Categorical features: ' + str(cat_attributes.columns))

    for column in cat_attributes:
        if column != 'income':
            sns.countplot(y=column, hue=data[label], data = cat_attributes)
            plt.show()
