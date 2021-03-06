import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import cohen_kappa_score
import FuzzyAlgorithm
from statistics import median

def prepare_data(name_and_position_of_file, is_header_present, name_or_number_of_target_column,
                 separator, percent_of_test_examples, is_oversampling_enabled):
    if is_header_present:
        df = pd.read_csv(name_and_position_of_file, sep=separator)
        y = df[name_or_number_of_target_column].values
        df = df.drop(name_or_number_of_target_column, axis=1)
    else:
        df = pd.read_csv(name_and_position_of_file, header=None, sep=separator)
        y_classification = df.columns[name_or_number_of_target_column - 1]
        y = df[y_classification].values
        df = df.drop(y_classification, axis=1)
    df = df.fillna(value=df.mean())
    df_norm = (df - df.mean()) / (df.max() - df.min())
    X = df_norm.values
    Xx, Xt, yy, yt = train_test_split(X, y, test_size=percent_of_test_examples, stratify=y)
    if is_oversampling_enabled:
        smt = SMOTE()
        Xx, yy = smt.fit_sample(Xx, yy)
        Xx, yy = shuffle(Xx, yy)
    return Xx, Xt, yy, yt

def find_categories(y):
    return list(set(y))

def match_categories(category, y):
    return [1 if _y == category else 0 for _y in y]

def find_min_max(X):
    return np.min(X, axis=0), np.max(X, axis=0)

def find_expanded_min_max(X, delta):
    min, max = find_min_max(X)
    return [i - delta for i in min], [i + delta for i in max]

def find_subset_by_category(X, y, category):
    return [x for x, _y in zip(X, y) if _y == category]

def RMSE(y_1, y_2):
    return sqrt(mean_squared_error(y_1, y_2))

def transform_parameters_to_indyvidual(list_of_parameters):
    indyvidual = []
    for col in list_of_parameters:
        indyvidual.extend(col)
    return indyvidual

def transform_indyvidual_to_parameters(indyvidual):
    number_of_parameters = 2
    list_of_parameters = []
    for i in range(len(indyvidual)//number_of_parameters):
        b = indyvidual[number_of_parameters*i]
        c = indyvidual[number_of_parameters*i + 1]
        list_of_parameters.append([b, c])
    return list_of_parameters

def accuracy(y_1, y_2):
    counter = 0
    for yy_1, yy_2 in zip(y_1, y_2):
        if yy_1 == yy_2:
            counter += 1
    return counter/len(y_1)

def find_avg_of_vectors_by_column(vectors):
    return np.mean(vectors, axis=0)

def plot_errors(errors, is_plot_saved, name_of_plot):
    epochs = range(1, len(errors[0]) + 1)
    fig = plt.figure()
    ax = plt.subplot()
    ax.plot(epochs, errors[0], "b-", label="Średni błąd na zbiorach trenujących")
    ax.plot(epochs, errors[1], "r-", label="Średni błąd na zbiorach testowych")
    plt.xlabel('iteracja', fontsize=15)
    plt.ylabel('Średnie RMSE', fontsize=15)
    ax.legend()
    if is_plot_saved:
        fig.savefig(name_of_plot, bbox_inches='tight')
    plt.show()

def plot_accuracies(accuracies, is_plot_saved, name_of_plot):
    epochs = range(1, len(accuracies[0]) + 1)
    fig = plt.figure()
    ax = plt.subplot()
    ax.plot(epochs, accuracies[0], "b-", label="Średnia celność predykcji na zbiorach trenujących")
    ax.plot(epochs, accuracies[1], "r-", label="Średnia celność predykcji na zbiorach testowych")
    plt.xlabel('iteracja', fontsize=15)
    plt.ylabel('Średnie wartość predykcji', fontsize=15)
    ax.legend()
    if is_plot_saved:
        fig.savefig(name_of_plot, bbox_inches='tight')
    plt.show()

def plot_scores(scores, is_plot_saved, name_of_plot):
    epochs = range(1, len(scores[0]) + 1)
    fig = plt.figure()
    ax = plt.subplot()
    ax.plot(epochs, scores[0], "b-", label="Średnia wartość współczynnika Kappa Cohena na zbiorach trenujących")
    ax.plot(epochs, scores[1], "r-", label="Średnia wartość współczynnika Kappa Cohena na zbiorach testowych")
    plt.xlabel('iteracja', fontsize=15)
    plt.ylabel('Średnie wartość współczynnika Kappa Cohena', fontsize=15)
    ax.legend()
    if is_plot_saved:
        fig.savefig(name_of_plot, bbox_inches='tight')
    plt.show()

def MMC(y_1, y_2):
    return matthews_corrcoef(y_1, y_2)

def CKS(y_1, y_2):
    return cohen_kappa_score(y_1, y_2)

def run_test(name_and_position_of_file, is_header_present, name_or_number_of_target_column,
                                    separator, percent_of_test_examples, is_oversampling_enabled,
                                    number_of_iterations, is_plot_saved,
                                    name_of_saved_file, name_of_error_plot, name_of_accuracy_plot, name_of_score_plot):
    all_train_errors = []
    all_test_errors = []
    all_train_accuracies = []
    all_train_scores = []
    all_test_accuracies = []
    all_test_scores = []
    for i in range(1, number_of_iterations + 1):
        Xx, Xt, yy, yt = prepare_data(name_and_position_of_file, is_header_present, name_or_number_of_target_column,
                                            separator, percent_of_test_examples, is_oversampling_enabled)
        parameters_and_categories, train_errors, test_errors = FuzzyAlgorithm.learn_system(Xx, yy, Xt, yt)
        predictions = [FuzzyAlgorithm.run_system(Xx, [[rule[0][i], rule[1], rule[2], rule[3]] for rule in parameters_and_categories])
                       for i in range(len(parameters_and_categories[0][0]))]
        train_accuracies_for_epochs = [accuracy(predicion, yy) for predicion in predictions]
        train_scores_for_epochs = [CKS(predicion, yy) for predicion in predictions]
        predictions = [FuzzyAlgorithm.run_system(Xt, [[rule[0][i], rule[1], rule[2], rule[3]] for rule in parameters_and_categories])
                       for i in range(len(parameters_and_categories[0][0]))]
        test_accuracies_for_epochs = [accuracy(predicion, yt) for predicion in predictions]
        test_scores_for_epochs = [CKS(predicion, yt) for predicion in predictions]
        all_train_accuracies.append(train_accuracies_for_epochs)
        all_train_scores.append(train_scores_for_epochs)
        all_test_accuracies.append(test_accuracies_for_epochs)
        all_test_scores.append(test_scores_for_epochs)
        all_train_errors.append(find_avg_of_vectors_by_column(train_errors))
        all_test_errors.append(find_avg_of_vectors_by_column(test_errors))
    avg_all_train_accuracies = find_avg_of_vectors_by_column(all_train_accuracies)
    avg_all_train_scores = find_avg_of_vectors_by_column(all_train_scores)
    avg_all_test_accuracies = find_avg_of_vectors_by_column(all_test_accuracies)
    avg_all_test_scores = find_avg_of_vectors_by_column(all_test_scores)
    avg_all_train_erros = find_avg_of_vectors_by_column(all_train_errors)
    avg_all_test_errors = find_avg_of_vectors_by_column(all_test_errors)
    min_acc, max_acc = find_min_max([xx[-1] for xx in all_test_accuracies])
    median_acc = median([xx[-1] for xx in all_test_accuracies])
    min_score, max_score = find_min_max([xx[-1] for xx in all_test_scores])
    median_score = median([xx[-1] for xx in all_test_scores])
    with open(name_of_saved_file, "w") as file:
        file.write("Min of acc = {} Median of acc = {} Max of acc = {} \n".format(min_acc, median_acc, max_acc))
        file.write("Min of score = {} Median of score = {} Max of score = {} \n".format(min_score, median_score, max_score))
        file.write("Train avg errors = {} \n".format(avg_all_train_erros))
        file.write("Test avg errors = {} \n".format(avg_all_test_errors))
        file.write("Train avg accuracies = {} \n".format(avg_all_train_accuracies))
        file.write("Test avg accuracies = {} \n".format(avg_all_test_accuracies))
        file.write("Train avg scores = {} \n".format(avg_all_train_scores))
        file.write("Test avg scores = {} \n".format(avg_all_test_scores))
    file.close()
    plot_accuracies([avg_all_train_accuracies, avg_all_test_accuracies], is_plot_saved, name_of_accuracy_plot)
    plot_scores([avg_all_train_scores, avg_all_test_scores], is_plot_saved, name_of_score_plot)
    plot_errors([avg_all_train_erros, avg_all_test_errors], is_plot_saved, name_of_error_plot)