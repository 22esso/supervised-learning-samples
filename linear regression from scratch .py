import math
import random

import numpy
import numpy as np
import matplotlib.pyplot as plt

# splitting the data in 80% (Train) / 20% (Test)
def Train_test_split(X , Y):
    X_Train = X[:int(len(X)*0.8)]
    Y_Train = Y[:int(len(X) * 0.8)]
    X_Test = X[int(len(X)*0.8):]
    Y_Test = Y[int(len(X) * 0.8):]

    return X_Train,Y_Train,X_Test,Y_Test

# running the line equation given the factors
def predict(X,m,c):
    Y_hat = X*m +c
    return Y_hat

# calculating the Mean square error after the prediction
def MSE(Y,Y_hat):
    error_squared = sum(((Y-Y_hat) ** 2) )/ len(Y)
    return error_squared

def GD(X_train,Y_train,m,c,learning_rate):
    for i in range(10000):
        Y_hat = predict(X_train,m,c)
        Dm =(numpy.float64(-2)/len(X_train)) * X_train.T.dot(Y_train-Y_hat)
        Dc = (numpy.float64(-2)/len(X_train)) * sum((Y_train-Y_hat))
        m = m - (learning_rate * Dm)
        c = c - (learning_rate * Dc)
        #print(MSE(Y_train,Y_hat))
    return m,c

if __name__ == '__main__':
    # creating numbers from 0 to 199
    x = np.arange(200)
    # creating a set of number that can be used as bias for th Y value
    delta = np.random.uniform(-50,30,size=(200,))
    # creating the y coordinate values
    y = .4 * x + 3 + delta
    # splitting the data
    X_Train,Y_Train,X_Test,Y_Test = Train_test_split(x,y)

    # initializing the slope and the bias
    m = 1
    c = 1
    learning_rate = 0.0001
    Y_hat = predict(X_Train, m, c)
    print('initial error : ',MSE(Y_hat,Y_Train))

    m,c = GD(X_Train,Y_Train,m,c,learning_rate)

    print('optimal Slope value :' + str(m))
    print('optimal constant value :' + str(c))
    Y_hat = predict(X_Test, m, c)
    print('the error for the test sample : ',MSE(Y_Test,Y_hat))

    plt.plot(X_Train,predict(X_Train,m,c), color='red')

    #plotting it
    plt.scatter(x,y)
    plt.show()