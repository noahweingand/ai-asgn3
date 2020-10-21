import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
import math
import seaborn as sns
import numpy as np

iterations = 50000

day_1 = pd.read_csv("./train_data_1.txt", header=None)
day_2 = pd.read_csv("./train_data_2.txt", header=None)
day_3 = pd.read_csv("./train_data_3.txt", header=None)
day_4 = pd.read_csv("./test_data_4.txt", header=None)

def normalize_data(data):
    values = data.values
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
    if quad is True:
        values = np.column_stack((hours, hours_squared, consumptions))
        squared_df = pd.DataFrame(values)
        squared_df.columns = ['HOURS', 'QUAD HOURS', 'CONSUMPTION']
        squared_df[['BIAS']] = 1
        return (squared_df[['HOURS', 'QUAD HOURS', 'BIAS']].to_numpy(), squared_df[['CONSUMPTION']].to_numpy())
    if cube is True:
        hours_cubed = hours**3
        values = np.column_stack((hours, hours_squared, hours_cubed, consumptions))
        cubed_df = pd.DataFrame(values)
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

    return weights, train_error

def graph_results(title, dataset, plot_arr):
    sns.set(rc={'figure.figsize': (10, 8)})
    sns.set_style("whitegrid")
    sns.scatterplot(data=dataset, x='HOURS',
                    y='CONSUMPTION', linewidth=0)
  
    for params in plot_arr:
        x, y, graph_label, graph_color = params
        plt.plot(x, y, label=graph_label, color=graph_color)

    plt.legend(loc='best', borderaxespad=0.)
    plt.title(title)
    plt.ylim(-.1, 1.1)
    plt.xlim(-.1, 1.1)
    plt.show()

all_days = [normalize_data(day_1), normalize_data(day_2), normalize_data(day_3)]
test_norm = normalize_data(day_4)

def get_test_error(dataset, weights, archtype):
    input_x = dataset[['HOURS']].to_numpy()
    actual = dataset[["CONSUMPTION"]].to_numpy()
    predictions = []
    if(archtype == "linear"):
        x_weight = weights[0]
        bias = weights[1]
        for i in range(len(input_x)):
            x = input_x[i]
            pred = (x_weight * x) + bias
            predictions.append(pred)
    if(archtype == "quadratic"):
        x_weight1 = weights[0]
        x_weight2 = weights[1]
        bias = weights[2]
        for i in range(len(input_x)):
            x = input_x[i]
            pred = (x_weight2 * x**2) + (x_weight1 * x) + bias
            predictions.append(pred)
    if(archtype == "cubic"):
        x_weight1 = weights[0]
        x_weight2 = weights[1]
        x_weight3 = weights[2]
        bias = weights[3]
        for i in range(len(input_x)):
            x = input_x[i]
            pred = (x_weight3 * x**3) + (x_weight2 * x**2) + (x_weight1 * x) + bias
            predictions.append(pred)

    test_error = 0
    for i in range(len(actual)):
        error = actual[i] - predictions[i]
        test_error = test_error + math.pow(error, 2)

    return test_error

# the number of samples to plot for the line
num_samples = 100

# to store all x, y for each architecture
linear_plot = []
quad_plot = []
cubic_plot = []

linear_weights = []
quad_weights = []
cubic_weights = []

#training error
linear_te = []
quad_te = []
cubic_te = []
# ARCHITECTURE 1 - x

for (i, day) in enumerate(all_days):
    in_arr, out_arr = get_input_output(day)
    weights, error = train(in_arr, out_arr, 0.3, [0.5, 0.5])
    x_weight = weights[0][0]
    bias = weights[1][0]
    x = np.linspace(-1, 2, num_samples)
    y = (x_weight * x) + bias
    linear_plot.append((x, y, 'linear', 'green'))
    linear_weights.append((x_weight, bias))
    linear_te.append(error)

# ARCHITECTURE 2 - x^2

for (i, day) in enumerate(all_days):
    in_arr, out_arr = square_or_cube(day, quad=True, cube=False)
    weights, error = train(in_arr, out_arr, 0.1, [0.5, 0.5, 0.5])
    x_weight1 = weights[0][0]
    x_weight2 = weights[1][0]
    bias = weights[2][0]
    x = np.linspace(-1, 2, num_samples)
    y = (x_weight2 * x**2) + (x_weight1 * x) + bias
    quad_plot.append((x, y, 'quad', 'blue'))
    quad_weights.append((x_weight1, x_weight2, bias))
    quad_te.append(error)

# ARCHITECTURE 3 - x^3

for (i, day) in enumerate(all_days):
    in_arr, out_arr = square_or_cube(day, quad=False, cube=True)
    weights, error = train(in_arr, out_arr, 0.05, [0.5, 0.5, 0.5, 0.5])
    x_weight1 = weights[0][0]
    x_weight2 = weights[1][0]
    x_weight3 = weights[2][0]
    bias = weights[3][0]
    x = np.linspace(-1, 2, num_samples)
    y = (x_weight3 * x**3) + (x_weight2 * x**2) + (x_weight1 * x) + bias
    cubic_plot.append((x, y, 'cubic', 'red'))
    cubic_weights.append((x_weight1, x_weight2, x_weight3, bias))
    cubic_te.append(error)

#Training Error
for i in range(3):
    print("[TRAIN ERROR: DAY %d Linear]\t%f" %(i+1, linear_te[i]))
    print("[TRAIN ERROR: DAY %d Quadratic]\t%f" %(i+1, quad_te[i]))
    print("[TRAIN ERROR: DAY %d Cubic]\t%f" %(i+1, cubic_te[i]))

# Training Plots
graph_results("DAY 1", all_days[0], [linear_plot[0], quad_plot[0], cubic_plot[0]])
graph_results("DAY 2", all_days[1], [linear_plot[1], quad_plot[1], cubic_plot[1]])
graph_results("DAY 3", all_days[2], [linear_plot[2], quad_plot[2], cubic_plot[2]])

#average linear weights
avg_linear_weights = np.divide(np.add(linear_weights[0], np.add(linear_weights[1], linear_weights[2])), np.array([3,3]))
#avgerage quadratic weights
avg_quad_weights = np.divide(np.add(quad_weights[0], np.add(quad_weights[1], quad_weights[2])), np.array([3,3,3]))
#average cubic weights
avg_cubic_weights = np.divide(np.add(cubic_weights[0], np.add(cubic_weights[1], cubic_weights[2])), np.array([3,3,3,3]))


linear_avg_plot = []
quad_avg_plot = []
cubic_avg_plot = []
#linear plot using average weights
x_weight = avg_linear_weights[0]
bias = avg_linear_weights[1]
x = np.linspace(-1, 2, num_samples)
y = (x_weight * x) + bias
linear_avg_plot.append((x, y, 'linear', 'green'))

#quad plot using average weights
x_weight1 = avg_quad_weights[0]
x_weight2 = avg_quad_weights[1]
bias = avg_quad_weights[2]
x = np.linspace(-1, 2, num_samples)
y = (x_weight2 * x**2) + (x_weight1 * x) + bias
quad_avg_plot.append((x, y, 'quad', 'blue'))


#cubic plot using average weights 
x_weight1 = avg_cubic_weights[0]
x_weight2 = avg_cubic_weights[1]
x_weight3 = avg_cubic_weights[2]
bias = avg_cubic_weights[3]
x = np.linspace(-1, 2, num_samples)
y = (x_weight3 * x**3) + (x_weight2 * x**2) + (x_weight1 * x) + bias
cubic_avg_plot.append((x, y, 'cubic', 'red'))


#Testing Error
# linear_test_error = get_test_error(test_norm, linear_weights[0], 'linear')
# quad_test_error = get_test_error(test_norm, quad_weights[0], 'quadratic')
# cubic_test_error = get_test_error(test_norm, cubic_weights[0], 'cubic')
linear_test_error = get_test_error(test_norm, avg_linear_weights, 'linear')
quad_test_error = get_test_error(test_norm, avg_quad_weights, 'quadratic')
cubic_test_error = get_test_error(test_norm, avg_cubic_weights, 'cubic')

print("[TEST ERROR Linear]\t%f" %(linear_test_error))
print("[TEST ERROR Quadratic]\t%f" %(quad_test_error))
print("[TEST ERROR Cubic]\t%f" %(cubic_test_error))

# Testing
# graph_results("DAY 4 linear", test_norm, [linear_plot[0]])
# graph_results("DAY 4 quad", test_norm, [quad_plot[0]])
# graph_results("DAY 4 cubic", test_norm, [cubic_plot[0]])
graph_results("DAY 4 linear", test_norm, linear_avg_plot)
graph_results("DAY 4 quad", test_norm, quad_avg_plot)
graph_results("DAY 4 cubic", test_norm, cubic_avg_plot)

