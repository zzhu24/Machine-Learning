import NN
import data_loader
import perceptron
import numpy as np
import matplotlib.pyplot as plt


train = data_loader.load_mnist_data()

folds=[]
l=len(train)
fold_size=l/5
for i in range(5):
    if i<4:
        folds.append(train[i*fold_size:(i+1)*fold_size])
    else:
        folds.append(train[i*fold_size:l])
        result_training=[]
    for i in range(5):
        current=[]
        for j in range(5):
            if i!=j:
                current+=folds[j]
        result_training.append([current,folds[i]])


parameters=[]
result=[]
for a in [10,50,100]:
    for b in [0.1,0.01]:
        for c in ['relu','tanh']:
            for d in [10,50]:
                parameters.append(['mnist',a,b,c,d])


para_size = len(parameters)
for i in range(para_size):
    accuracy=0
    for j in range(5):
        net=NN.create_NN(parameters[i][0],parameters[i][1],parameters[i][2],parameters[i][3],parameters[i][4])
        net.train(result_training[j][0])
        accuracy+=net.evaluate(result_training[j][1])
        result.append(accuracy/5)
for i in range(len(result)):
    print "domain=",parameters[i][0],",", "batch size=",parameters[i][1],",", "learning rate=",parameters[i][2],",", "activation function=",parameters[i][3],",", "hidden layer width=",parameters[i][4],",", "accuracy:",result[i]
best=np.argmax(result)
print "best parameter and accuracy:"
print "domain=",parameters[best][0],",", "batch size=",parameters[best][1],",", "learning rate=",parameters[best][2],",", "activation function=",parameters[best][3],",", "hidden layer width=",parameters[best][4],",", "accuracy:",result[best]



