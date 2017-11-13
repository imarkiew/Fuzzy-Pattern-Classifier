import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.utils import shuffle

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
    else:
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
    number_of_parameters = 3
    list_of_parameters = []
    for i in range(len(indyvidual)//number_of_parameters):
        r = indyvidual[number_of_parameters*i]
        p = indyvidual[number_of_parameters*i + 1]
        q = indyvidual[number_of_parameters*i + 2]
        list_of_parameters.append([r, p, q])
    return list_of_parameters

def accuracy(y_1, y_2):
    counter = 0
    for yy_1, yy_2 in zip(y_1, y_2):
        if yy_1 == yy_2:
            counter += 1
    return (counter/len(y_1))*100