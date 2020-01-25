

from input.heart_import import get_heart_data
from sklearn.preprocessing import StandardScaler

from als import ALS
import pandas as pd

if __name__ == '__main__':  # to avoid multiprocessor children to begin from start
    data = get_heart_data()

    scaler = StandardScaler()
    data.iloc[:, 1:] = scaler.fit_transform(data.iloc[:, 1:]) # TODO: do this in ALS object instead

    accuracies = []
    repetitions = 10
    n_points_to_add_at_a_time = 5

    for i in range(repetitions):
        print('##################################################################')
        print('REPETITION NUMBER: ' + str(i))
        als = ALS(unsplit_data=data,
                  learning_method='bayesian_random',
                  n_points_labeled_delete=50,
                  n_points_labeled_keep=10,
                  scale=False,
                  n_points_to_add_at_a_time=n_points_to_add_at_a_time,
                  seed=i)
        als.run_experiment()
        accuracies.append(als.accuracies)

    #print(accuracies)
    a = pd.DataFrame(accuracies).mean(axis=0)
    print(a)
    a.to_csv('bayesian_accuracy_k10_d50_heart_rep' + str(repetitions)
             + '_step' + str(n_points_to_add_at_a_time) + '.csv')
