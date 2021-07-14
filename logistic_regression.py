# apt-get install python3-matplotlib
# python3 logistic_regression.py
import matplotlib.pyplot as plt
import numpy as np
import random

#generate random data
x1 = []
x2 = []
y = []
for i in range(100):
    x1.append(random.uniform(0, 5))
    x2.append(random.uniform(0, 3))
    y.append(0)

for i in range(100):
    x1.append(random.uniform(3, 10))
    x2.append(random.uniform(4, 10))
    y.append(1)

#array to keep errors to show its plot
errors = []
b0 = 0   
b1 = 0       
b2 = 0       
alpha = 0.01 
e = 2.71828
err = 0

# training:
for i in range(500000):
    idx = i % 200
    if idx == 0:
        if err != 0:
            errors.append(err)
        err = 0
    
    gX = (b0 + b1 * (x1[idx]) + b2 * (x2[idx]))
    hX = 1 / (1 + pow(e, -gX))
    err += pow(hX - y[idx], 2)
    
    b0 -= alpha * (hX - y[idx])
    b1 -= alpha * (hX - y[idx]) * x1[idx]
    b2 -= alpha * (hX - y[idx]) * x2[idx]

plt.figure(0)

#validation:
for i in range(len(y)):
    gX = (b0 + b1 * (x1[i]) + b2 * (x2[i]))
    hX = 1 / (1 + pow(e, -gX))
    if hX > 0.5:
        plt.scatter(x1[i], x2[i], s=100, c='green')
    else:
        plt.scatter(x1[i], x2[i], s=100, c='red')

x_var = np.linspace(0,10,10000)
y_var = (-b0 - b1*x_var) / b2 

#just for curiosity - the linear function should always point to 0.5
for i in range(len(y)):
    gX = (b0 + b1 * (x_var[i]) + b2 * (y_var[i]))
    hX = 1 / (1 + pow(e, -gX))
    print("should be 0.5:", hX)
    
plt.plot(x_var, y_var, '-r', label='')

plt.figure(1)
x_var = np.linspace(0,500000,len(errors))
plt.plot(x_var, errors, '-r', label='')

plt.grid()
plt.show()
