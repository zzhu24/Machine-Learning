import NN
import data_loader
import perceptron
import numpy as np
import matplotlib.pyplot as plt





train= data_loader.load_mnist_data()
test = data_loader.load_circle_data()


len(train[0][0])

result=[]
net = NN.create_NN('circle', 10, 0.1, 'relu', 10)
NN_curve=net.train_with_learning_curve(train)
NN=[]
for tuple in NN:
    NN.append(tuple[1])
result.append(net.evaluate(test))
plt.plot(np.arange(1,100,1),NN,label='NN')



perceptron = perceptron.Perceptron(len(train[0][0]))
curve_training=perceptron.train_with_learning_curve(train)
perceptron=[]
for tuple in curve_training:
    perceptron.append(tuple[1])
result.append(perceptron.evaluate(test))


plt.plot(np.arange(1,100,1),NN,label='NN')
plt.plot(np.arange(1,100,1),np.multiply(perceptron,100),label='perceptron')