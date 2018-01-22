import Tools

# name_and_position_of_file = "./Data/reprocessed.hungarian.data"
# is_header_present = False
# name_or_number_of_target_column = 14
# separator = " "

name_and_position_of_file = "./Data/diabetes.csv"
is_header_present = True
name_or_number_of_target_column = "class"
separator = ","

percent_of_test_examples = 0.3
is_oversampling_enabled = True

number_of_iterations = 3
is_plot_saved = True
path = "./Results/"
name_of_error_plot = path + "errors.png"
name_of_accuracy_plot = path + "accuracies.png"
name_of_score_plot = path + "scores.png"
name_of_saved_file = path + "stat.csv"
Tools.run_test(name_and_position_of_file, is_header_present, name_or_number_of_target_column,
                                    separator, percent_of_test_examples, is_oversampling_enabled,
                                    number_of_iterations, is_plot_saved, name_of_saved_file,
                                    name_of_error_plot, name_of_accuracy_plot, name_of_score_plot)

