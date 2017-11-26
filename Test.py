import Tools
import FuzzyAlgorithms

# name_and_position_of_file = "./diabetes.csv"
# is_header_present = True
# name_or_number_of_target_column = "class"
# separator = ","

name_and_position_of_file = "./iris.data"
is_header_present = False
name_or_number_of_target_column = 5
separator = ","

# name_and_position_of_file = "./reprocessed.hungarian.data"
# is_header_present = False
# name_or_number_of_target_column = 14
# separator = " "

# name_and_position_of_file = "./ConvertedPeptidome2_240.csv"
# is_header_present = False
# name_or_number_of_target_column = 1
# separator = ","

# name_and_position_of_file = "./ConvertedLungCancer_500.csv"
# is_header_present = False
# name_or_number_of_target_column = 1
# separator = ","

percent_of_test_examples = 0.3
is_oversampling_enabled = False

Xx, Xt, yy, yt = Tools.prepare_data(name_and_position_of_file, is_header_present, name_or_number_of_target_column,
                                    separator, percent_of_test_examples, is_oversampling_enabled)
parameters_and_categories, train_and_test_errors = FuzzyAlgorithms.learn_system(Xx, yy, Xt, yt)
prediction = FuzzyAlgorithms.run_system(Xt, parameters_and_categories)
print(prediction)
accuracy = Tools.accuracy(prediction, yt)
print(accuracy)
avg_train_errors = Tools.find_avg_of_vectors_by_column(train_and_test_errors[0])
avg_test_errors = Tools.find_avg_of_vectors_by_column(train_and_test_errors[1])
Tools.plot_errors([avg_train_errors, avg_test_errors], True, "errors.png")
