import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
import math
import seaborn as sns
import numpy as np

iterations = 5000
learn = 0.1

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
    data[['BIAS']] = 1
    return (data[['HOURS', 'BIAS']].to_numpy(), data[['CONSUMPTION']].to_numpy())

def train_linear(input_arr, alpha, epsilon, gain):
    weights = [0.5, 0.5]
    scaled_data = normalize_data(input_arr)
    input_arr, output_arr = get_input_array(scaled_data)
    total_error = 0
    #print(output_arr)

    for i in range(iterations):
        total_error = 0
        for j in range(len(input_arr)):
            net = 0
            for k in range(len(weights)):
                net = net + input_arr[j][k]*weights[k]
            #out = soft_activation(net, gain)
            error = output_arr[j] - net
            total_error = total_error + math.pow(error, 2)
            learn = alpha * error
            print(i, j, net, error, learn, weights, total_error)
            for z in range(len(weights)):
                weights[z] = weights[z] + (learn * input_arr[j][z])
        train_error = total_error
        # if(total_error < epsilon):
        #     break
    return weights #, train_error

def graph_results(title, dataset, final_weights):
    sns.set(rc={'figure.figsize': (10, 8)})
    sns.set_style("whitegrid")
    sns.scatterplot(data=dataset, x='HOURS',
                    y='CONSUMPTION', linewidth=0)
    xf_weight = final_weights[0][0]
    f_bias = final_weights[1][0]

    print("xf_weight: %s" % xf_weight)
    print("f_bias: %s" % f_bias)

    # fm = (-1 * xf_weight) / (yf_weight)
    # fb = (-1 * f_bias) / yf_weight
    fx = np.linspace(-1, 2, 50)
    fy = (xf_weight * fx) + f_bias
    plt.plot(fx, fy, label="Final Line", color='green')

    plt.legend(loc='best', borderaxespad=0.)
    plt.title(title)
    plt.ylim(-.1, 1.1)
    plt.xlim(-.1, 1.1)
    plt.show()



norm_day_1 = normalize_data(day_1)
norm_day_2 = normalize_data(day_2)
norm_day_3 = normalize_data(day_3)

# ARCHITECTURE 1 - x
day_1_trained = train_linear(norm_day_1, 0.3, 50, 0.2)
day_2_trained = train_linear(norm_day_2, 0.3, 50, 0.2)
day_3_trained = train_linear(norm_day_3, 0.3, 50, 0.2)
graph_results("DAY 1", norm_day_1, day_1_trained)
graph_results("DAY 2", norm_day_2, day_2_trained)
graph_results("DAY 3", norm_day_3, day_3_trained)


# ARCHITECTURE 2 - x^2


# ARCHITECTURE 3 - x^3