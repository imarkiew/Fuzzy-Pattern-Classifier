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

number_of_iterations = 10
is_plot_saved = True
name_of_error_plot = "errors.png"
name_of_accuracy_plot = "accuracies.png"
name_of_score_plot = "scores.png"
name_of_saved_file = "stat.csv"
Tools.run_test(name_and_position_of_file, is_header_present, name_or_number_of_target_column,
                                    separator, percent_of_test_examples, is_oversampling_enabled,
                                    number_of_iterations, is_plot_saved, name_of_saved_file,
                                    name_of_error_plot, name_of_accuracy_plot, name_of_score_plot)

