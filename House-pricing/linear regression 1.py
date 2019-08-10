#linear regression

import numpy as np
import matplotlib.pyplot as plt

# One step of gradient descent
def step_gradient(b_current, w_current, train_x, train_y, learning_rate):
    grad_b = 0
    grad_w = 0
    #J = 0
    m = len(train_y)
    
    for i in range(m):
        x = train_x[i]
        y = train_y[i]
        #cost = (1/2*m) * (((w_current * x) + b_current) - y)**2
        #J = J - learning_rate*cost
        grad_b += (1/m) * (((w_current * x) + b_current) - y)
        grad_w += (1/m) * x * (((w_current * x) + b_current) - y)
        b_new = b_current - (learning_rate * grad_b)
        w_new = w_current - (learning_rate * grad_w)
        
    return [b_new, w_new]


def run_descent(train_x, train_y, init_b, init_w, epochs, learning_rate):
    b = init_b
    w = init_w
    for i in range(epochs):
        b, w = step_gradient(b, w, train_x, train_y, learning_rate)
    return [b, w]



def drawPlot(train_x, train_y, w, b):
    plt.plot(train_x, train_y, 'go')
    plt.plot([0, 7000], [0 + b, 7000*w + b], color='b', linestyle='-', linewidth=2)
    plt.xlabel('Size')
    plt.ylabel('Price')
    plt.tight_layout()
    plt.show()

#prediction function
def predict(w, b, x_new):
    predict = w*x_new + b
    return [predict]


# Our main function for set up everything and run linear regression
def run():
    file = 'houses.csv'
    points = np.array(np.genfromtxt(file, delimiter=',',skip_header=1))
    print(points[:])
                      
    learning_rate = 0.0000001  # I've tried .01 - .000001  but only this worked properly

    train_x = points[:,0]  # sizes of the houses
    #print(train_x)
    train_y = points[:,1]  # prices
    #print(points.shape())
    init_b = 0
    init_w = 0

    print('{} - number of training examples'.format(len(train_y)))
    print('k = 0, b = 0 | initial parameters')

    epochs = 20
    [b, w] = run_descent(train_x, train_y, init_b, init_w, epochs, learning_rate)

    #print('k = %.2f, b = %.2f | final parameters' % (w, b))
    print('k = {}, b = {} | final parameters'.format(w, b))
    #plot.drawPlot(train_x, train_y, w, b)
    m = len(train_y)
    print("prediction:", predict(w,b,1400))
    drawPlot(train_x, train_y, w, b)


if __name__ == '__main__':
    run()
