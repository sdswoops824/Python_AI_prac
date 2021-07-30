# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
from sklearn import preprocessing


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

input_data = np.array([[2.1, -1.9, 5.5],
                       [-1.5, 2.4, 3.5],
                       [0.5, -7.9, 5.6],
                       [5.9, 2.3, -5.8]])
print("\nOriginal array: \n", input_data)

# Binarizarion
data_binarized = preprocessing.Binarizer(threshold=0.5).transform(input_data)
print("\nBinarized Data:\n", data_binarized)


# Mean Removal
# display mean and std
print("Mean = ", input_data.mean(axis = 0))
print("Std deviation = ", input_data.std(axis = 0))

# remove mean and std
data_scaled = preprocessing.scale(input_data)
print("New_Mean = ", data_scaled.mean(axis =0))
print("New_Std deviation = ", data_scaled.std(axis = 0))

# min max scaling
data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0,1))
data_scaled_minmax = data_scaler_minmax.fit_transform(input_data)
print("\nMin max scaled data:\n", data_scaled_minmax)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
