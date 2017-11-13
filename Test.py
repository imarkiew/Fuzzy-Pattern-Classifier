import Tools
import FuzzyAlgorithms

name_and_position_of_file = "./diabetes.csv"
is_header_present = True
name_or_number_of_target_column = "class"
separator = ","

# name_and_position_of_file = "./iris.data"
# is_header_present = False
# name_or_number_of_target_column = 5
# separator = ","

# name_and_position_of_file = "./reprocessed.hungarian.data"
# is_header_present = False
# name_or_number_of_target_column = 14
# separator = " "

percent_of_test_examples = 0.3
is_oversampling_enabled = True

Xx, Xt, yy, yt = Tools.prepare_data(name_and_position_of_file, is_header_present, name_or_number_of_target_column,
                                    separator, percent_of_test_examples, is_oversampling_enabled)
parameters_and_categories = FuzzyAlgorithms.learn_system(Xx, yy)
prediction = FuzzyAlgorithms.run_system(Xt, parameters_and_categories)
print(prediction)
accuracy = Tools.accuracy(prediction, yt)
print(accuracy)