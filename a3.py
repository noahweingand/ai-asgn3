import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing

learn_rate = 0.1

day_1 = pd.read_csv("./train_data_1.txt")
day_2 = pd.read_csv("./train_data_2.txt")
day_3 = pd.read_csv("./train_data_3.txt")
test_data = pd.read_csv("./test_data_4.txt")

def normalize_data(data):
    values = data.values
    min_max_scaler = preprocessing.MinMaxScaler()
    scaled_values = min_max_scaler.fit_transform(values)
    scaled_data = pd.DataFrame(scaled_values)
    scaled_data.columns = ['HOURS', 'CONSUMPTION']

    return scaled_data

def get_input_array(data):
    data[['Bias']] = 1
    return data[['HOURS', 'CONSUMPTION', 'BIAS']].to_numpy()

norm_day_1 = normalize_data(day_1)
norm_day_2 = normalize_data(day_2)
norm_day_3 = normalize_data(day_3)

# ARCHITECTURE 1


# ARCHITECTURE 2


# ARCHITECTURE 3