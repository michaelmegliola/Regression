import numpy as np
import matplotlib.pyplot as plt
import time


class MyGraph:
    def __init__(self, x, y):
        plt.ion()
        self.count = 0
        self.figure = plt.figure()
        self.ax = self.figure.add_subplot(111)
        self.line1, = self.ax.plot(x, y, 'o')
        self.line2, = self.ax.plot([0], [0], 'b-')
        self.x_curve = np.arange(-10, 10, 0.1)
        plt.ylim(-8, 20)

    def update(self, a, b, c, error):
        plt.title('{:0d}'.format(self.count) + ' a=' + '{:.3f}'.format(a) + ' b=' + '{:.3f}'.format(b) + ' c=' + '{:.3f}'.format(
            c) + ' error=' + '{:.3f}'.format(error))
        self.line1.set_ydata(y)
        yi = [a * (xi ** 2) + b * xi + c for xi in self.x_curve]
        self.line2.set_xdata(self.x_curve)
        self.line2.set_ydata(yi)
        self.figure.canvas.draw()
        self.count += 1


x = [-4, -2, 0, 2, 4]
y = [17, 5, 1, 5, 17]

graph = MyGraph(x, y)

a = np.random.rand() * 5 - 10
b = np.random.rand() * 5 - 10
c = np.random.rand() * 5 - 10

alpha = 0.001
prior_error = 0

for i in range(10000):

    # these values are computed as the sums of series, so start at zero
    error = 0
    derror_da = 0
    derror_db = 0
    derror_dc = 0

    for j in range(len(x)):
        # ------------------------------------------------------------------
        #
        #        error = (prediction - observation)^2
        #
        #       |--------------------------------------------|
        #       | to apply the chain rule... the outer       |
        #       | function is (prediction - observation)^2   |
        #       |         |========prediction========| |obsv||
        error += np.square(a * x[j]**2 + b * x[j] + c - y[j])

        #           |====derivative of outer function ======| | w/r/t a |
        derror_da += 2 * (a * x[j]**2 + b * x[j] + c - y[j]) * x[j]**2

        #           |====derivative of outer function ======| | w/r/t b |
        derror_db += 2 * (a * x[j]**2 + b * x[j] + c - y[j]) * x[j]

        #           |====derivative of outer function ======| | w/r/t c |
        derror_dc += 2 * (a * x[j]**2 + b * x[j] + c - y[j]) * 1
        # ------------------------------------------------------------------

    # take a step in the right direction...
    a -= derror_da * alpha  # ...the direction is de/da
    b -= derror_db * alpha  # ...the direction is de/db
    c -= derror_dc * alpha  # ...the direction is de/dc

    graph.update(a, b, c, error)

    d_error = error - prior_error
    prior_error = error

    if abs(d_error) < 0.000001:
        time.sleep(120)
        break
