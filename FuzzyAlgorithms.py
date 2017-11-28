import numpy as np
import Tools
import GeneticAlgorithms

def pi_function(x, r, p, q, min, max):
    m = 2
    value = 0
    if min < x and p >= x:
        value = (2**(m - 1))*((x - min) / (r - min))**m
    elif p < x and r >= x:
        value = 1 - ((2**(m - 1))*((r - x) / (r - min))**m)
    elif r < x and q >= x:
        value = 1 - ((2 ** (m - 1)) * ((x - r) / (max - r)) ** m)
    elif q < x and max >= x:
        (2**(m - 1))*((max - x) / (max - r))**m
    return value

def aggregation_operator(values):
    return np.mean(values)

def aggregated_output(X, parameters, min, max):
    output = []
    for xx in X:
        values = []
        for col, min_col, max_col, param in zip(xx, min, max, parameters):
            values.append(pi_function(col, param[0], param[1], param[2], min_col, max_col))
        output.append(aggregation_operator(values))
    return output

def learn_system(X, y, Xt, yt):
    parameters_and_categories = []
    delta = 0.01
    cxpb = 0.5
    mutpb = 0.01
    start_population_size = 50
    size_of_offspring = 25
    number_of_epochs = 200
    categories = Tools.find_categories(y)
    hof_errors_for_categories = []
    test_errors_for_categories = []
    for category in categories:
        print("Learning for category {}".format(category))
        train_subset = Tools.find_subset_by_category(X, y, category)
        train_min, train_max = Tools.find_expanded_min_max(train_subset, delta)
        train_y_bin = Tools.match_categories(category, y)
        test_y_bin = Tools.match_categories(category, yt)
        indyvidual, train_and_test_errors = GeneticAlgorithms.run_genetic_algorithm(X, train_y_bin, Xt, test_y_bin, delta, train_min, train_max, cxpb, mutpb, start_population_size,
                                                size_of_offspring, number_of_epochs)
        print("\n")
        parameters_and_categories.append([Tools.transform_indyvidual_to_parameters(indyvidual), train_min, train_max, category])
        hof_errors_for_categories.append(train_and_test_errors[0])
        test_errors_for_categories.append(train_and_test_errors[1])
    return parameters_and_categories, hof_errors_for_categories, test_errors_for_categories

def run_system(X, parameters_and_categories):
    prediction = []
    for xx in X:
        output = []
        for rule in parameters_and_categories:
            parameters = rule[0]
            min = rule[1]
            max = rule[2]
            output.append(aggregated_output([xx], parameters, min, max)[0])
        index = output.index(np.max(output))
        prediction.append(parameters_and_categories[index][3])
    return prediction







