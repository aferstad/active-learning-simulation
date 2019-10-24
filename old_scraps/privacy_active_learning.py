


# SET GLOBAL CONSTANTS:
FINAL_TEST_SET_FRACTION = 0.3
UNLABELED_FRACTION = 0.95
TRAIN_FRACTION = 0.75
DELETE_FRACTION = 0 # TODO: CHANGE THIS WHEN APPLYING PRIVACY
ADD_FRACTION = 0.5



def split_data_in_two(data, fraction, seed = 300):
    majority = data.sample(frac=max(fraction, 1-fraction),
                             random_state=seed)
    minority = data.drop(majority.index)

    return majority.copy, minority.copy


#def split_data_in_five(data, fraction)




# DEFINE FUNCTIONS
def split_method_final_test(data, seed=300):
    method_set = data.sample(frac=1 - FINAL_TEST_SET_FRACTION,
                             random_state=seed)
    final_test_set = data.drop(method_set.index)

    return method_set, final_test_set


def split_labeled_unlabeled(data, seed=200):
    labeled = data.sample(frac=1 - UNLABELED_FRACTION, random_state=seed)
    unlabeled = data.drop(labeled.index)

    return labeled, unlabeled


def split_train_test(data, seed=150):
    train = data.sample(frac=TRAIN_FRACTION, random_state=seed)
    test = data.drop(train.index)

    train_y = train.iloc[:, 0]
    train_x = train.drop(train.columns[0], axis=1)

    test_y = test.iloc[:, 0]
    test_x = test.drop(test.columns[0], axis=1)

    return train_x, train_y, test_x, test_y


def split_keep_delete(data, seed=100):
    keep = data.sample(frac=1 - DELETE_FRACTION, random_state=seed)
    delete = data.drop(keep.index)

    return keep, delete


def train_tree(data):
    train_x, train_y, test_x, test_y = split_train_test(data)

    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier(min_samples_leaf=5,
                                 min_impurity_decrease=0.000001)

    # Train Decision Tree Classifer
    clf = clf.fit(train_x, train_y)

    #Predict the response for test dataset
    predict_y = clf.predict(test_x)

    feature_names = train_x.columns
    class_names = train_y.unique()

    #plot_tree(clf, feature_names, class_names)
    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(test_y, predict_y))
    print(confusion_matrix(test_y, predict_y))
    plot_tree_ROC(clf, test_x, test_y)
    plot_tree(clf, feature_names, class_names)

    y_scores = clf.predict_proba(test_x)[:, 1]

    average_precision = average_precision_score(test_y, y_scores)
    print('Average precision-recall score: {0:0.2f}'.format(average_precision))

    precision, recall, _ = precision_recall_curve(test_y, y_scores)
    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({
        'step': 'post'
    } if 'step' in signature(plt.fill_between).parameters else {})
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
        average_precision))


def plot_tree(clf, feature_names, class_names):
    dot_data = StringIO()
    export_graphviz(clf,
                    out_file=dot_data,
                    filled=True,
                    rounded=True,
                    special_characters=True,
                    feature_names=feature_names,
                    class_names=['0', '1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('tree.png')
    display(Image(graph.create_png()))


def plot_tree_ROC(clf, test_x, test_y):
    prob_predictions = clf.predict_proba(test_x)
    print('AUC: ' + str(roc_auc_score(test_y, prob_predictions[:, 1])))

    fpr, tpr, _ = roc_curve(test_y, prob_predictions[:, 1])

    plt.clf()
    plt.plot(fpr, tpr)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC curve')
    plt.show()


def train_logistic_regression(data, prints=True):
    train_x, train_y, test_x, test_y = split_train_test(data)

    # TODO: add n-fold CV and then train on whole labeled set after parameters set
    # don't run CV after each additional point

    # possible to set class weight to combat skewed data
    # also possible to adjust regularization parameters

    lr = LogisticRegression()

    lr.fit(train_x, train_y)

    if prints:
        print('Coefficients: ', lr.coef_)
        print('Interecept: ', lr.intercept_)

        predict_y = lr.predict(test_x)
        print('Confusion matrix: ', '\n', confusion_matrix(test_y, predict_y))

    return lr


def label_uncertain_data(lr, unlabeled, number_of_new_points):
    unlabeled_x = unlabeled.drop(unlabeled.columns[0], axis=1)
    prob_predictions = lr.predict_proba(unlabeled_x)
    prob_for_1 = prob_predictions[:, 1]

    unlabeled['abs_uncertainty'] = np.abs(prob_for_1 - 0.5)
    #unlabeled.abs_uncertainty.plot.hist()

    new_data = unlabeled.sort_values(
        'abs_uncertainty').iloc[0:number_of_new_points]
    new_unlabeled = unlabeled.sort_values(
        'abs_uncertainty').iloc[number_of_new_points:len(unlabeled)]

    new_data = new_data.drop(columns='abs_uncertainty')
    new_unlabeled = new_unlabeled.drop(columns='abs_uncertainty')

    return new_data, new_unlabeled


def active_learning(prepared_data,
                    active_learning_method='uncertainty',
                    seeds=[200, 150, 100]):

    # Splitting data
    method_data, final_test = split_method_final_test(prepared_data,
                                                      seed=seeds[0])
    labeled, unlabeled = split_labeled_unlabeled(method_data, seed=seeds[1])
    keep, delete = split_keep_delete(labeled, seed=seeds[2])

    # Train original model
    lr = LogisticRegression()
    current_lr = LogisticRegression()

    labeled_y = labeled.iloc[:, 0]
    labeled_x = labeled.drop(labeled.columns[0], axis=1)

    lr.fit(scale(labeled_x), labeled_y)
    current_lr.fit(scale(labeled_x), labeled_y)

    # Label new points:
    number_of_points_to_add = len(unlabeled) * ADD_FRACTION
    number_of_new_points = 0

    current_unlabeled = unlabeled.copy()
    current_labeled = labeled.copy()
    new_data = pd.DataFrame()

    if active_learning_method == 'uncertainty':

        while number_of_new_points < number_of_points_to_add:
            # This is very slow now because we train model again for each new point:
            step_size = 25
            number_of_new_points += step_size

            new_points, new_unlabeled = label_uncertain_data(
                current_lr, current_unlabeled, step_size)

            current_labeled = pd.concat([current_labeled, new_points], axis=0)
            current_unlabeled = new_unlabeled.copy()

            current_labeled_y = current_labeled.iloc[:, 0]
            current_labeled_x = current_labeled.drop(
                current_labeled.columns[0], axis=1)

            current_lr.fit(scale(current_labeled_x), current_labeled_y)

            new_data.append(new_points)

    # Code to get all data at once instead of one at a time:
    #new_data = label_uncertain_data(lr, unlabeled, number_of_new_points)
    else:
        print('active_learning_method not defined')

    # TODO: take 50 and 50 samples from different clusters
    # another method is to cluster data and then add one point from each cluster

    final_test_y = final_test.iloc[:, 0]
    final_test_x = final_test.drop(final_test.columns[0], axis=1)

    final_test_x = scale(final_test_x)

    original_predict = lr.predict(final_test_x)
    new_predict = current_lr.predict(final_test_x)

    original_F1 = f1_score(final_test_y, original_predict)
    new_F1 = f1_score(final_test_y, new_predict)

    # Calculate consistency
    keep_y = keep.iloc[:, 0]
    keep_x = keep.drop(keep.columns[0], axis=1)

    predict_original = lr.predict(keep_x)
    predict_new = current_lr.predict(keep_x)

    # TODO: also do this on TEST set instead of overlapping training set

    consistency = np.mean(np.abs(predict_original -
                                 predict_new))  # MEAN ABSOLUTE ERROR

    return original_F1, new_F1, consistency
