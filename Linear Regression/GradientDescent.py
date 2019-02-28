"""
Author: John Jacobson (u1201441)
Created: 2019-02-17

This is an implementation of Gradient Descent algorithms for use in linear regression learning for CS6350 at
University of Utah in Spring 2019.

    Coming soon

"""
import random
import math
import decimal


def gradient_descent(examples, labels, weight, batch_size, iterations, learning_constant):

    calc_weight = list(weight)
    r = learning_constant
    weights = [list(weight)]
    threshold = 0.000001
    converge = False

    for i in range(iterations):
        gradient = get_gradient(examples, labels, calc_weight, batch_size)
        prev_weight = list(calc_weight)
        for j in range(len(calc_weight)):
            calc_weight[j] = calc_weight[j] - r * gradient[j]

        weights.append(list(calc_weight))

        if math.sqrt(sum((decimal.Decimal(x-y))**2 for x,y in zip(prev_weight,calc_weight))) < threshold:
            converge = True

    return r, weights, converge


def get_gradient(examples, labels, weight, batch_size):
    dj = list(weight)

    if batch_size > 0:
        indices = random.sample(range(len(examples)), batch_size)
        batch = list([examples[i] for i in indices])
        batch_labels = list([labels[i] for i in indices])
    else:
        batch = list(examples)
        batch_labels = list(labels)

    for j in range(len(dj)):
        for i in range(len(batch)):
            dj[j] -= batch[i][j] * (batch_labels[i] - sum(x*y for x,y in zip(weight, batch[i])))

    return dj
