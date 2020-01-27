

from input.heart_import import get_heart_data
from sklearn.preprocessing import StandardScaler

from als import ALS
import pandas as pd

if __name__ == '__main__':  # to avoid multiprocessor children to begin from start
    data = get_heart_data()

    scaler = StandardScaler()
    data.iloc[:, 1:] = scaler.fit_transform(data.iloc[:, 1:]) # TODO: do this in ALS object instead
    consistencies = []
    accuracies = []
    repetitions = 10
    n_points_to_add_at_a_time = 5
    delete = 30
    keep = 10

    for i in range(repetitions):
        print('##################################################################')
        print('REPETITION NUMBER: ' + str(i))
        als = ALS(unsplit_data=data,
                  learning_method='bayesian_random',
                  n_points_labeled_delete=delete,
                  n_points_labeled_keep=keep,
                  scale=False,
                  n_points_to_add_at_a_time=n_points_to_add_at_a_time,
                  seed=i)
        als.run_experiment()
        consistencies.append(als.consistencies)
        accuracies.append(als.accuracies)

    #print(accuracies)
    a = pd.DataFrame(accuracies).mean(axis=0)
    c = pd.DataFrame(consistencies).mean(axis=0)
    print(a)
    print(c)
    save_path = '_bayesian_k' + str(keep) + '_d' + str(delete) + '_heart_rep' + str(repetitions)+ '_step' + str(n_points_to_add_at_a_time) + '.csv'

    a.to_csv('accuracy' + save_path, header = True)
    c.to_csv('consistency' + save_path, header = True)

