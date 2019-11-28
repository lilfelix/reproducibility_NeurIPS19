"""
This module performs classification on top of the learned representations of the time series in the datasets. It
basically uses the SVM classifier using the scikit-learn module: https://scikit-learn.org/stable/modules/svm.html
For cross validation and examples read this: https://scikit-learn.org/stable/modules/cross_validation.html
the number of folds is left as 5 (suggested by sklearn warning), the default scoring of SVC (accuracy) is used
read about the iid parameter in GridSearchCV documentation.
"""
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import numpy as np
import math
import csv
import os


def perform_svm_all(rep_folder, save_csv=False, recompute_all=False):
    """
    :param rep_folder: the folder containing the representations. Path should be relative to the 'data/' folder. For
    instance, it could be 'feature 2' indicating the path 'data/features 2'.
    :param save_csv:
    :param recompute_all
    :return:
    """

    # reading the csv file for the datasets whose results we have computed already
    if not recompute_all:
        with open('../results/features_red_purple_29oct.csv', 'rt') as f:
            reader = csv.reader(f)
            csv_list = list(reader)
            # no need to consider datasets whose results are already computed
            already_computed = [row[0] for row in csv_list]
            # print('Already computed datasets are:', already_computed)

    all_datasets_acc = []

    # getting the names of all the datasets sorted by name
    datasets_names = sorted(next(os.walk('../data/{}'.format(rep_folder)))[1])
    print("In [perform_svm_all]: found {} datasets in folder data/{}".format(len(datasets_names), rep_folder))

    # just put it here in order not to screw up the code (should be removed later)
    avoid_list = []
    no_cv_datasets = ['Adiac', 'ECG5000', 'DiatomSizeReduction', 'FaceFour', 'FiftyWords',
                      'InsectEPGSmallTrain', 'Phoneme', 'OliveOil', 'Symbols', 'WordSynonyms',
                      'PigAirwayPressure', 'Fungi', 'PigArtPressure', 'PigCVP']

    # vary length and/or not normalized (waiting for the new list) - includes Felix's normalizing labels
    # avoid_list = ['AllGestureWiimoteX', 'AllGestureWiimoteY', 'AllGestureWiimoteZ',
    #               'DodgerLoopGame', 'DodgerLoopWeekend', 'GestureMidAirD1', 'GestureMidAirD2',
    #               'GestureMidAirD3', 'GesturePebbleZ1', 'GesturePebbleZ2', 'PickupGestureWiimoteZ', 'PLAID',
    #               'ShakeGestureWiimoteZ', 'Fungi', 'Rock', 'EOGVerticalSignal', 'EOGHorizontalSignal',
    #               'Chinatown', 'SemgHandSubjectCh2', 'SemgHandGenderCh2', 'SemgHandMovementCh2',
    #               'GunPointMaleVersusFemale', 'PigArtPressure', 'PigCVP', 'GunPointOldVersusYoung',
    #               'GunPointAgeSpan', 'HouseTwenty']


    # not sure if they are included above
    # nan_list = ['Chinatown', 'LargeKitchenAppliances',
    #             'Lightning2', 'Lightning7', 'SmallKitchenAppliances', 'EOGHorizontalSignal', 'EOGVerticalSignal',
    #             'Fungi', 'GunPointAgeSpan', 'GunPointMaleVersusFemale', 'GunPointOldVersusYoung', 'HouseTwenty',
    #             'PigAirwayPressure',
    #             'PigArtPressure',
    #             'PigCVP',
    #             'Rock',
    #             'SemgHandGenderCh2',
    #             'SemgHandMovementCh2',
    #             'SemgHandSubjectCh2']

    # this list has no effect at the moment (could be removed)
    # long_list = ['EthanolLevel', 'ElectricDevices', 'Crop', 'FordB', 'FordA', 'NonInvasiveFetalECGThorax1',
    #               'NonInvasiveFetalECGThorax2', 'PhalangesOutlinesCorrect', 'MelbournePedestrian',
    #               'EthanolLevel', 'StarLightCurves', 'Wafer',
    #               'ECG5000',
    #               'TwoPatterns',
    #               'ChlorineConcentration',
    #               'UWaveGestureLibraryAll',
    #               'UWaveGestureLibraryX',
    #               'UWaveGestureLibraryY',
    #               'UWaveGestureLibraryZ',
    #               'Yoga',
    #               'FreezerRegularTrain',
    #               'FreezerSmallTrain',
    #               'MelbournePedestrian',
    #               'MixedShapesRegularTrain',
    #               'MixedShapesSmallTrain',
    #               'Mallat',
    #               'FacesUCR',
    #               'Wine']

    # bad_list = avoid_list + long_list

    for dataset in datasets_names:
        # no need to consider datasets whose results are already computed
        if not recompute_all:  # probably these two consecutive if's could be coded better
            if dataset in already_computed:
                print('Skipping {}... Already computed.'.format(dataset))
                continue

        # checking for nan (should be removed after Sofia's checking)
        if dataset in avoid_list:
            print('Skipping', dataset, '... because it is in bad list', '\n')
            continue

        print('Performing on dataset:', dataset)
        dataset_acc = [dataset]  # putting the name of the dataset at first (used in creating the CSV file)
        train_acc_lst, test_acc_lst = [], []
        combined_train_rep, combined_test_rep = None, None

        # if the dataset has limited samples per class, C = inf when training the SVM
        c_inf = True if dataset in no_cv_datasets else False

        # doing classification for each possible k
        for k in [1, 2, 5, 10]:
            print('\nPerforming k={}...'.format(k))
            train_acc, test_acc, train_rep, train_labels, test_rep, test_labels = \
                perform_svm_on_representations(rep_folder, dataset, k, c_inf, return_representations=True)

            # to be removed
            if k == 1:
                print('In [perform_svm_all]: some labels:', train_labels[:10])

            # dataset_acc.append(round(test_acc, 3))
            train_acc_lst.append(round(train_acc, 3))
            test_acc_lst.append(round(test_acc, 3))

            # concatenating representations for the corresponding k
            if combined_train_rep is None:
                combined_train_rep = train_rep
                combined_test_rep = test_rep
            else:
                combined_train_rep = np.concatenate((combined_train_rep, train_rep), axis=1)  # concatenating along column
                combined_test_rep = np.concatenate((combined_test_rep, test_rep), axis=1)

        # print(combined_train_rep.shape, combined_test_rep.shape, train_labels.shape, test_labels.shape)

        # train_labels and test_labels is the same regardless of value of k so we use them directly
        print('\nPerforming k=combined...')  # to be removed
        combined_train_acc, combined_test_acc = \
            perform_svm_combined(combined_train_rep, train_labels, combined_test_rep, test_labels, c_inf)

        train_acc_lst.append(round(combined_train_acc, 3))
        test_acc_lst.append(round(combined_test_acc, 3))

        dataset_acc += train_acc_lst + test_acc_lst
        print('dataset {} acc:'.format(dataset), dataset_acc, '\n')
        print('=' * 50)

        all_datasets_acc.append(dataset_acc)

        # saving the accuracy results for the corresponding dataset
        if save_csv:
            with open('../results/features_red_purple_29oct.csv', 'a') as f:
                w = csv.writer(f, dialect='excel')
                w.writerow(dataset_acc)
    # print(all_datasets_acc)


def perform_svm_combined(combined_train_rep, train_labels, combined_test_rep, test_labels, c_inf):
    """
    To be completed.
    :return:
    """
    classifier = SVMClassifier(combined_train_rep, train_labels, c_inf)  # train the combined classifier
    train_acc = classifier.classify(combined_train_rep, train_labels)
    test_acc = classifier.classify(combined_test_rep, test_labels)  # getting test accuracy
    return train_acc, test_acc


def perform_svm_on_representations(rep_folder, dataset, k, c_inf, return_representations=False):
    """
    This function tests the performance of the SVM classifier trained on the representations. The default path for the
    representations is considers to be '../data/features'.
    :param dataset (str): the dataset whose representations are to be read
    :param k (int): number of negative samples
    :return: the test accuracy of the trained classifier
    """
    train_path = '../data/{}/{}/{}/k_{}.npz'.format(rep_folder, dataset, 'train', k)
    test_path = '../data/{}/{}/{}/k_{}.npz'.format(rep_folder, dataset, 'test', k)

    # loading the representations and labels
    train_features, test_features = np.load(train_path), np.load(test_path)
    train_rep, train_labels = train_features['features'], train_features['labels']
    test_rep, test_labels = test_features['features'], test_features['labels']
    # print("In [svm_classification]: read file with shapes: train {}, {} and test {}, {}".
          # format(train_rep.shape, train_labels.shape, test_rep.shape, test_labels.shape))

    # getting the trained SVM with optimized C value using cross-validation
    classifier = SVMClassifier(train_rep, train_labels, c_inf)
    train_acc = classifier.classify(train_rep, train_labels)
    test_acc = classifier.classify(test_rep, test_labels)

    if return_representations:
        return train_acc, test_acc, train_rep, train_labels, test_rep, test_labels
    else:
        return train_acc, test_acc


def svm_cross_validation(train_rep, y_train, cv, iid, grid, refit, verbose=True):
    """
    For more info about the parameters and documentation see the comment at the beginning of the file.
    :param train_rep: np array of shape (n_samples, rep_length), training time series representations
    :param y_train: np array of shape (n_samples,), train set labels
    :param cv: number of cross validation folds
    :param iid: see the GridSearchCV documentation
    :param grid: the parameters grid space
    :param refit: if the SVM with the best parameters should be refit to the whole train set
    :param verbose: if Tru, details of cross validation results are printed
    :return: a GridSearchCV object containing the SVM with the best parameters
    """
    print('In [svm_cross_validation]: doing cross validation...')
    clf = GridSearchCV(estimator=SVC(), cv=cv, iid=iid, refit=refit, param_grid=grid)
    clf.fit(train_rep, y_train)  # call fit() with all param configurations

    if verbose:  # part of the code adapted from the above-mentioned link
        print("In [svm_cross_validation]: best parameters set found with best score: %0.3f" % clf.best_score_)
        print("Best params found:", clf.best_params_, '\n')
        print("In [svm_cross_validation]: grid scores on development set:")
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    return clf


class SVMClassifier:
    def __init__(self, train_rep, y_train, c_inf):
        """
        This basically initializes the classier. The type of the kernel used is 'rbf'. The gamma value of thr RBF
        functions is not discussed in the paper, so we leave it as default. The penalty value C for misclassification
        should be optimized using cross-validation, but is infinity for datasets of small size or small data points
        per class (see Appendix S1.2).

        Currently I use the params mentioned in the paper so the constructor does not accept any param itself.
        Could be edited later.
        """

        # no cross validation for small datasets
        if c_inf:  # C_inf for datasets with limited data samples per class
            print('In [SVMClassifier]: Limited dataset... No cross validation')
            self.classifier = SVC(C=math.inf, kernel='rbf', gamma='auto')
            self.classifier.fit(train_rep, y_train)

        # performing cross validation
        else:  # see svm_cross_validation() comments regarding the following params
            c_range = [pow(10, i) for i in np.arange(-4, 5, dtype='float32')]  # 10 ** i where i is in [-4, 4]
            cv = 5
            iid = True
            grid = [{'kernel': ['rbf'], 'gamma': ['auto'], 'C': c_range}]
            refit = True

            # returns the classifier with the optimized params (C in our case), and then refit on the whole train data
            # Note that here classifier is a GridSearchCV object and not SVC
            self.classifier = svm_cross_validation(train_rep, y_train, cv, iid, grid, refit, verbose=False)

    def classify(self, test_rep, y_test):
        """
        Classifying the test representations.
        :param test_rep: np array of shape (n_samples, rep_length), where rep_length is the representation length
        :param y_test: np array of shape (n_samples,)
        :return: test accuracy
        """
        # y_pred = self.classifier.predict(test_rep)
        test_acc = self.classifier.score(test_rep, y_test)
        return test_acc
