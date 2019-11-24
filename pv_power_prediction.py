#!/home/heping/anaconda3/envs/py3/bin/python3

import pandas as pd
import numpy as np
import os
import tensorflow as tf

# Read dataset from training csv files
def read_file(file_path):
    if True == os.path.exists(file_path):
        if True == os.path.isdir(file_path):
            print("The file " + str(file_path) + " is directory! Please input csv file!")
            return -1
        else:
            try:
                fileRead = pd.read_csv(file_path)
                data = fileRead.values
                return data
            except IOError as e:
                print(e)
    else:
        print(str(file_path) + " does not exit! Please check it!")
        pass
"""
# Read all csv files in directory path
def read_file_from_path(dir_path):
    if True == os.path.exists(dir_path):
        if True == os.path.isdir(dir_path):
            all_data = []
            file_list = os.listdir(dir_path)
            print("In " + str(dir_path) + ", there exists " + str(file_list))
            i = 0
            for file_name in file_list:
                file_path = dir_path + file_name
                file_data = read_file(file_path)
                if 0 == i:
                    all_data = file_data
                    i = i + 1
                all_data = np.vstack((all_data, file_data))
            return all_data
        else:
            print(str(dir_path) + " is not directory! Please check it again!")
            pass
    else:
        print(str(dir_path) + " does not exist! Please check it again!")
        pass
"""
def get_train_data(file_path):
    train_data = read_file(file_path)
    train_x = train_data[:, 1:7]
    train_y = train_data[:, 8].reshape(-1, 1)
    return train_x, train_y

def get_test_data(file_name):
    test_data = read_file(file_name)
    test_x = test_data[:, 2:]
    return test_x

def split_train_data(train_x_raw, train_y_raw, train_scale):
    all_size = train_x_raw.shape[0]
    train_size = int(all_size * train_scale)
    validation_size = all_size - train_size
    train_x = train_x_raw[0:train_size, :]
    validation_x = train_x_raw[train_size:-1, :]
    train_y = train_y_raw[0:train_size, :]
    validation_y = train_y_raw[train_size:-1, :]
    return train_x, train_y, validation_x, validation_y

def get_all_data():
    current_path = os.getcwd()
    train_path = current_path + "/dataset/train/"
    test_path = current_path + "/dataset/test/"

    train_file_list = []
    test_file_list = []
    train_x_raw_list = []
    train_y_raw_list = []
    test_x_list = []
    train_x_list = []
    train_y_list = []
    validation_x_list = []
    validation_y_list = []

    for i in range(1, 11):
        train_file_name = train_path + "train_" + str(i) + ".csv"
        test_file_name = test_path + "test_" + str(i) + ".csv"

        train_x_raw, train_y_raw = get_train_data(train_file_name)
        test_x= get_test_data(test_file_name)

        train_x, train_y, validation_x, validation_y = split_train_data(train_x_raw, train_y_raw, 0.8)

        train_file_list.append(train_file_name)
        test_file_list.append(test_file_name)
        train_x_raw_list.append(train_x_raw)
        train_y_raw_list.append(train_y_raw)
        test_x_list.append(test_x)
        train_x_list.append(train_x)
        train_y_list.append(train_y)
        validation_x_list.append(validation_x)
        validation_y_list.append(validation_y)

    return train_x_list, train_y_list, validation_x_list, validation_y_list, test_x_list

"""
def normalize_dataset(dataset_list):
    normalized_dataset_list = []
    for dataset in dataset_list:
        normalized_dataset = (dataset - np.mean(dataset, axis = 0)) / np.std(dataset, axis = 0)
        normalized_dataset_list.append(normalized_dataset)
    return normalized_dataset_list
"""

def extract_valid_dataset(x_list, y_list):
    valid_x_list = []
    valid_y_list = []
    installed_capacity_list = [20, 30, 10, 20, 21, 10, 40, 30, 50, 20]
    for i in range(len(y_list)):
        valid_x = []
        valid_y = []
        for num in range(len(y_list[i])):
            if y_list[i][num] >= installed_capacity_list[i] * 0.03:
                valid_x.append(x_list[i][num, :])
                valid_y.append(y_list[i][num])
        valid_x = np.array(valid_x).reshape(-1, x_list[i].shape[1])
        valid_y = np.array(valid_y).reshape(-1, y_list[i].shape[1])
        valid_x_list.append(valid_x)
        valid_y_list.append(valid_y)
    return valid_x_list, valid_y_list

# Calculate mae (mean absolute error a day)
def calculate_mae(actual_power_list, prediction_power_list):
    installed_capacity_list = [20, 30, 10, 20, 21, 10, 40, 30, 50, 20]
    mae_list = []
    for i in range(len(prediction_power_list)):
        effective_power_num_list = [num for num in range(len(actual_power_list[i])) if actual_power_list[i][num] >= installed_capacity_list[i] * 0.03]
        power_sum = 0
        for num in effective_power_num_list:
            power_sum = power_sum + np.abs(actual_power_list[i][num] - prediction_power_list[i][num])
        mae = power_sum / installed_capacity_list[i] / len(effective_power_num_list)
        mae_list.append(mae)
    return mae_list

def build_model(train_x, train_y, validation_x, validation_y, test_x):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units = 32,
                              activation = 'relu',
                              kernel_initializer = 'random_uniform',
                              bias_initializer = 'zeros',
                              #kernel_regularizer = tf.keras.regularizers.l2(0.01),
                              input_shape = (train_x.shape[1],)),
        #tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units = 32, activation = 'relu'),
        #tf.keras.layers.Dropout(0.2),
        #tf.keras.layers.Dense(units = 32, activation = 'relu'),
        #tf.keras.layers.Dropout(0.2),
        #tf.keras.layers.Dense(units = 128, activation = 'relu'),
        tf.keras.layers.Dense(units = 16,
                              activation = 'relu',
                              #kernel_regularizer = tf.keras.regularizers.l2(0.01),
                              kernel_initializer = 'random_uniform',
                              bias_initializer = 'zeros'),
        #tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units = 1)
    ])
    model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mae'])
    history = model.fit(train_x,
                        train_y,
                        epochs = 20,
                        batch_size = 64,
                        verbose = 2,
                        validation_data = (validation_x, validation_y))

    prediction = model.predict(test_x, batch_size = 32)
    return history, prediction

def output_results_to_file(prediction_list, file_name):
    all_predictions = prediction_list[0]
    for prediction_now in prediction_list[1:]:
        all_predictions = np.vstack((all_predictions, prediction_now))
    all_predictions = all_predictions.reshape(-1)
    numbers = range(1, len(all_predictions) + 1)
    dataframe = pd.DataFrame({'id':numbers, 'prediction':all_predictions})
    dataframe.to_csv(file_name, index = False, sep = ',')

def main():
    train_x_list, train_y_list, validation_x_list, validation_y_list, test_x_list = get_all_data()
    train_x_list, train_y_list = extract_valid_dataset(train_x_list, train_y_list)
    validation_x_list, validation_y_list = extract_valid_dataset(validation_x_list, validation_y_list)
    
    history_list = []
    prediction_list = []
    train_mae_list = []
    train_loss_list = []
    validation_mae_list = []
    validation_loss_list = []

    for i in range(len(train_x_list)):
        history, prediction = build_model(train_x_list[i], train_y_list[i], validation_x_list[i], validation_y_list[i], validation_x_list[i])
        history_list.append(history)
        train_mae_list.append(history.history['mean_absolute_error'])
        train_loss_list.append(history.history['loss'])
        validation_mae_list.append(history.history['val_mean_absolute_error'])
        validation_loss_list.append(history.history['val_loss'])
        prediction_list.append(prediction)

    """
    history, prediction = build_model_0(train_x_list[0], train_y_list[0], validation_x_list[0], validation_y_list[0], validation_x_list[0])
    history_list.append(history)
    train_mae_list.append(history.history['mean_absolute_error'])
    train_loss_list.append(history.history['loss'])
    validation_mae_list.append(history.history['val_mean_absolute_error'])
    validation_loss_list.append(history.history['val_loss'])
    prediction_list.append(prediction)

    history, prediction = build_model_1(train_x_list[1], train_y_list[1], validation_x_list[1], validation_y_list[1], validation_x_list[1])
    history_list.append(history)
    train_mae_list.append(history.history['mean_absolute_error'])
    train_loss_list.append(history.history['loss'])
    validation_mae_list.append(history.history['val_mean_absolute_error'])
    validation_loss_list.append(history.history['val_loss'])
    prediction_list.append(prediction)

    history, prediction = build_model_2(train_x_list[2], train_y_list[2], validation_x_list[2], validation_y_list[2], validation_x_list[2])
    history_list.append(history)
    train_mae_list.append(history.history['mean_absolute_error'])
    train_loss_list.append(history.history['loss'])
    validation_mae_list.append(history.history['val_mean_absolute_error'])
    validation_loss_list.append(history.history['val_loss'])
    prediction_list.append(prediction)

    history, prediction = build_model_3(train_x_list[3], train_y_list[3], validation_x_list[3], validation_y_list[3], validation_x_list[3])
    history_list.append(history)
    train_mae_list.append(history.history['mean_absolute_error'])
    train_loss_list.append(history.history['loss'])
    validation_mae_list.append(history.history['val_mean_absolute_error'])
    validation_loss_list.append(history.history['val_loss'])
    prediction_list.append(prediction)

    history, prediction = build_model_4(train_x_list[4], train_y_list[4], validation_x_list[4], validation_y_list[4], validation_x_list[4])
    history_list.append(history)
    train_mae_list.append(history.history['mean_absolute_error'])
    train_loss_list.append(history.history['loss'])
    validation_mae_list.append(history.history['val_mean_absolute_error'])
    validation_loss_list.append(history.history['val_loss'])
    prediction_list.append(prediction)

    history, prediction = build_model_5(train_x_list[5], train_y_list[5], validation_x_list[5], validation_y_list[5], validation_x_list[5])
    history_list.append(history)
    train_mae_list.append(history.history['mean_absolute_error'])
    train_loss_list.append(history.history['loss'])
    validation_mae_list.append(history.history['val_mean_absolute_error'])
    validation_loss_list.append(history.history['val_loss'])
    prediction_list.append(prediction)

    history, prediction = build_model_6(train_x_list[6], train_y_list[6], validation_x_list[6], validation_y_list[6], validation_x_list[6])
    history_list.append(history)
    train_mae_list.append(history.history['mean_absolute_error'])
    train_loss_list.append(history.history['loss'])
    validation_mae_list.append(history.history['val_mean_absolute_error'])
    validation_loss_list.append(history.history['val_loss'])
    prediction_list.append(prediction)

    history, prediction = build_model_7(train_x_list[7], train_y_list[7], validation_x_list[7], validation_y_list[7], validation_x_list[7])
    history_list.append(history)
    train_mae_list.append(history.history['mean_absolute_error'])
    train_loss_list.append(history.history['loss'])
    validation_mae_list.append(history.history['val_mean_absolute_error'])
    validation_loss_list.append(history.history['val_loss'])
    prediction_list.append(prediction)

    history, prediction = build_model_8(train_x_list[8], train_y_list[8], validation_x_list[8], validation_y_list[8], validation_x_list[8])
    history_list.append(history)
    train_mae_list.append(history.history['mean_absolute_error'])
    train_loss_list.append(history.history['loss'])
    validation_mae_list.append(history.history['val_mean_absolute_error'])
    validation_loss_list.append(history.history['val_loss'])
    prediction_list.append(prediction)

    history, prediction = build_model_9(train_x_list[9], train_y_list[9], validation_x_list[9], validation_y_list[9], validation_x_list[9])
    history_list.append(history)
    train_mae_list.append(history.history['mean_absolute_error'])
    train_loss_list.append(history.history['loss'])
    validation_mae_list.append(history.history['val_mean_absolute_error'])
    validation_loss_list.append(history.history['val_loss'])
    prediction_list.append(prediction)
    """

    
    mae_list = calculate_mae(validation_y_list, prediction_list)
    print(mae_list)
    

    #prediction_list_test = [np.array([[1, 1, 1]]), np.array([[2, 2, 2]]), np.array([[3, 3, 3]])]
    #print(prediction_list_test[1].shape)
    output_results_to_file(prediction_list, "predictions.csv")
    output_results_to_file(train_mae_list, "train_mae.csv")
    output_results_to_file(validation_mae_list, "validation_mae.csv")
    output_results_to_file(train_loss_list, "train_loss.csv")
    output_results_to_file(validation_loss_list, "validation_loss.csv")
     

if __name__ == "__main__":
    main()
"""
    temp1 = [np.array([[0.01, 2, 3], [2, 5, 0.003], [11, 2, 33], [2, 33, 1]]), np.array([[1, 0.002, 4], [2, 0.01, 6], [22, 1, 23], [2, 21, 11]])]
    print(temp1)
    result_temp = normalize_dataset(temp1)
    print(result_temp)
    print(result_temp * np.std(result_temp, axis = 0) + np.mean(result_temp, axis = 0))
"""


"""
    train_x_list = normalize_dataset(train_x_raw_list)
    train_y_list = normalize_dataset(train_y_raw_list)
    validation_x_list = normalize_dataset(validation_x_raw_list)
    validation_y_list = normalize_dataset(validation_y_raw_list)
    test_x_list = normalize_dataset(test_x_raw_list)
"""
