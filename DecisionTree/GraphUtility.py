

import matplotlib.pyplot as plt


def graph(data, file_name, x_label, y_label):

    data_legend = []

    for sublist in data:
        plt.plot(sublist[0])
        data_legend.append(sublist[1])

    plt.legend(data_legend, loc='upper right')
    plt.title(file_name)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()












