import numpy as np
import matplotlib.pyplot as plt
import time


class MyGraph:
    def __init__(self, x, y):
        plt.ion()
        self.figure = plt.figure()
        self.ax = self.figure.add_subplot(111)
        self.line1, = self.ax.plot(x, y, 'o')
        self.line2, = self.ax.plot(x, y, 'b-')
        self.count = 0

    def update(self, m, b, error):
        plt.title('{:0d}'.format(self.count) + ' m=' + '{:.3f}'.format(m) + ' b=' + '{:.3f}'.format(b) + ' error=' + '{:.3f}'.format(error))
        self.line1.set_ydata(y)
        yi = [m*xi+b for xi in x]
        self.line2.set_ydata(yi)
        self.figure.canvas.draw()
        self.count += 1


x = [0, 1, 2, 3, 4, 5]
y = [0, 2, 1, 7, 4, 12]

graph = MyGraph(x, y)

m = np.random.rand() * 5 - 10
b = np.random.rand() * 5 - 10

alpha = 0.001
prior_error = 0

for i in range(10000):

    # these values are computed as the sums of series, so start at zero
    error = 0
    derror_dm = 0
    derror_db = 0

    for j in range(len(x)):
        # ------------------------------------------------------------------
        #
        #        error = (prediction - observation)^2
        #
        #       |--------------------------------------------|
        #       | to apply the chain rule... the outer       |
        #       | function is (prediction - observation)^2   |
        #       |         |=prediction=| |obsv|              |
        error += np.square(m * x[j] + b - y[j])

        #           |=derivative of outer fn=|  | w/r/t m|
        derror_dm += 2 * (m * x[j] + b - y[j]) * x[j]

        #           |=derivative of outer fn=|  | w/r/t b|
        derror_db += 2 * (m * x[j] + b - y[j]) * 1

    # take a step in the right direction...
    m -= derror_dm * alpha  # the direction is de/dm
    b -= derror_db * alpha  # the direction is de/db

    graph.update(m, b, error)

    d_error = error - prior_error
    prior_error = error

    if abs(d_error) < 0.000001:
        time.sleep(120)
        break
