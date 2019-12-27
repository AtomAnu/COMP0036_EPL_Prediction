import os
import pandas as pd

def load_file(file_name):
    """
    Load .csv file using pd.read_csv
    :param file_name: string containing the file name
    :return: extracted and stripped data in the DataFrame type
    """

    #find the path of the wanted file
    file_dir = os.path.dirname(os.path.abspath(__file__))
    for root, dirs, files in os.walk(file_dir):
        for name in files:
            if name == file_name:
                file_path = os.path.join(root, name)

    #load the wanted file using pd.read_csv
    loaded_data = pd.read_csv(file_path)

    #remove blank columns
    data_columns = list(loaded_data)
    for col in data_columns:
        if 'Unnamed' not in col:
            last_col_idx = data_columns.index(col)
    data_stripped = loaded_data.iloc[:, :last_col_idx + 1]

    return data_stripped

def format_data(training_file_name,test_file_name):
    """
    Format the loaded training and test data. Date column is formatted. Columns with categorical features are converted
    to multiple columns with the binary format.
    :param training_file_name: string containing the training file name
    :param test_file_name: string containing the test file name
    :return: formatted training and test data in the DataFrame type
    """

    #load training and test files
    training_data_stripped = load_file(training_file_name)
    test_data_stripped = load_file(test_file_name)

    #convert all columns categorical features into multiple columns with the binary format
    #training and test data are combined
    training_and_test_binary = pd.get_dummies(pd.concat([training_data_stripped.drop(columns=['Date']),
                                                         test_data_stripped.drop(columns=['Date'])], sort=False))

    #extract the training data
    training_binary = training_and_test_binary.iloc[:len(training_data_stripped),:]

    #extract the test data
    test_binary = training_and_test_binary.iloc[len(training_data_stripped):,:]
    test_binary_columns = [col for col in training_and_test_binary
                           for original_col in list(test_data_stripped) if col.startswith(original_col)]
    test_binary = test_binary[test_binary_columns]


    #format the Date column
    training_date_formatted = pd.DataFrame(pd.to_datetime(training_data_stripped.iloc[:,0]))
    test_date_formatted = pd.DataFrame(pd.to_datetime(test_data_stripped.iloc[:,0]))

    #combine the Date column with the rest
    training_data_formatted = pd.concat([training_date_formatted, training_binary], axis=1)
    test_data_formatted = pd.concat([test_date_formatted, test_binary], axis=1)

    return training_data_formatted, test_data_formatted
