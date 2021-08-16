import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def read_data(input_file):
    input_data = np.loadtxt(input_file, delimiter = None)
    dates = pd.date_range('1950-01', periods=input_data.shape[0], freq='M')
    output = pd.Series(input_data[:, 2], index=dates) # 1 or 2
    return output

if __name__ == '__main__':
    input_file = "C:/Users/swagata/OneDrive - Hiroshima University/personal/Job/Python_AI_practice/Python_AI_prac/AO.txt"
    timeseries = read_data(input_file)

    plt.figure()
    timeseries.plot()
    plt.show()

# slicing time series data
    timeseries['1980':'1990'].plot()
    plt.show()