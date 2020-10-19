import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
import math
import seaborn as sns
import numpy as np

iterations = 500
learn = 0.1

day_1 = pd.read_csv("./train_data_1.txt", header=None)
day_2 = pd.read_csv("./train_data_2.txt", header=None)
day_3 = pd.read_csv("./train_data_3.txt", header=None)
test_data = pd.read_csv("./test_data_4.txt", header=None)

def normalize_data(data):
    values = data.values
    print(values)
    min_max_scaler = preprocessing.MinMaxScaler()
    scaled_values = min_max_scaler.fit_transform(values)
    scaled_data = pd.DataFrame(scaled_values)
    scaled_data.columns = ['HOURS', 'CONSUMPTION']

    return scaled_data

def get_input_output(data):
    data[['BIAS']] = 1
    return (data[['HOURS', 'BIAS']].to_numpy(), data[['CONSUMPTION']].to_numpy())

def square_or_cube(data, quad=False, cube=False):
    values = data.values
    hours = values[:, 0]
    hours_squared = hours**2
    consumptions = values[:, 1]
    min_max_scaler = preprocessing.MinMaxScaler()
    if quad is True:
        values = np.column_stack((hours, hours_squared, consumptions))
        scaled_values = min_max_scaler.fit_transform(values)
        squared_df = pd.DataFrame(scaled_values)
        squared_df.columns = ['HOURS', 'QUAD HOURS', 'CONSUMPTION']
        squared_df[['BIAS']] = 1
        return (squared_df[['HOURS', 'QUAD HOURS', 'BIAS']].to_numpy(), squared_df[['CONSUMPTION']].to_numpy())
    if cube is True:
        hours_cubed = hours**3
        values = np.column_stack((hours, hours_squared, hours_cubed, consumptions))
        scaled_values = min_max_scaler.fit_transform(values)
        cubed_df = pd.DataFrame(scaled_values)
        cubed_df.columns = ['HOURS', 'QUAD HOURS', 'CUBE HOURS', 'CONSUMPTION']
        cubed_df[['BIAS']] = 1
        return (cubed_df[['HOURS', 'QUAD HOURS', 'CUBE HOURS', 'BIAS']].to_numpy(), cubed_df[['CONSUMPTION']].to_numpy())

def train(input_arr, output_arr, alpha, weights):
    total_error = 0

    for i in range(iterations):
        total_error = 0

        for j in range(len(input_arr)):
            net = 0
            for k in range(len(weights)):
                net = net + input_arr[j][k]*weights[k]
            error = output_arr[j] - net
            total_error = total_error + math.pow(error, 2)
            learn = alpha * error
            # print(i, j, net, error, learn, weights, total_error)
            for z in range(len(weights)):
                weights[z] = weights[z] + (learn * input_arr[j][z])

        train_error = total_error

    return weights 

def graph_results(title, dataset, x, y):
    sns.set(rc={'figure.figsize': (10, 8)})
    sns.set_style("whitegrid")
    sns.scatterplot(data=dataset, x='HOURS',
                    y='CONSUMPTION', linewidth=0)
    # xf_weight = final_weights[0][0]
    # f_bias = final_weights[1][0]

    # print("xf_weight: %s" % xf_weight)
    # print("f_bias: %s" % f_bias)

    # fm = (-1 * xf_weight) / (yf_weight)
    # fb = (-1 * f_bias) / yf_weight
    # fx = np.linspace(-1, 2, 50)
    # fy = (xf_weight * fx) + f_bias
    plt.plot(x, y, label="Final Line", color='green')

    plt.legend(loc='best', borderaxespad=0.)
    plt.title(title)
    plt.ylim(-.1, 1.1)
    plt.xlim(-.1, 1.1)
    plt.show()


all_days = [normalize_data(day_1), normalize_data(day_2), normalize_data(day_3)]

# linear_weights

# ARCHITECTURE 1 - x

for (i, day) in enumerate(all_days):
    in_arr, out_arr = get_input_output(day)
    weights = train(in_arr, out_arr, 0.3, [0.5, 0.5])
    x_weight = weights[0][0]
    bias = weights[1][0]
    x = np.linspace(-1, 2, 50)
    y = (x_weight * x) + bias
    graph_results("DAY " + str(i + 1) + " linear", day, x, y)

# ARCHITECTURE 2 - x^2

for (i, day) in enumerate(all_days):
    in_arr, out_arr = square_or_cube(day, quad=True, cube=False)
    print("")
    print(in_arr, out_arr)
    weights = train(in_arr, out_arr, 0.3, [0.5, 0.5, 0.5])
    x_weight1 = weights[0][0]
    x_weight2 = weights[1][0]
    bias = weights[2][0]
    x = np.linspace(-1, 2, 50)
    y = (x_weight2 * x**2) + (x_weight1 * x) + bias
    graph_results("DAY " + str(i + 1) + " quadratic", day, x, y)

# ARCHITECTURE 3 - x^3

for (i, day) in enumerate(all_days):
    in_arr, out_arr = square_or_cube(day, quad=False, cube=True)
    weights = train(in_arr, out_arr, 0.3, [0.5, 0.5, 0.5, 0.5])
    x_weight1 = weights[0][0]
    x_weight2 = weights[1][0]
    x_weight3 = weights[2][0]
    bias = weights[3][0]
    x = np.linspace(-1, 2, 50)
    y = (x_weight3 * x**3) + (x_weight2 * x**2) + (x_weight1 * x) + bias
    graph_results("DAY " + str(i + 1) + " quadratic", day, x, y)