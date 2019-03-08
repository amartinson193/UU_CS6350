"""
Author: John Jacobson (u1201441)
Created: 2019-03-02

This file contains data presentation utilities, mostly functions for creating charts.

    Coming soon

"""

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












