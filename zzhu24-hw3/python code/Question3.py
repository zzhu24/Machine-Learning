import numpy as np
import matplotlib.pyplot as plt

from add_noise import add_noise
from gen import gen
from Perceptron import Perceptron
from Perceptron_Margin import Perceptron_Margin
from Winnow import Winnow
from Winnow_Margin import Winnow_Margin
from AdaGrad import AdaGrad

"""
Function that test the algoritem and plot for question 3
"""








#m = 100
(train_y,train_x)=gen(10,100,1000,50000,True)
(test_y,test_x)=gen(10,100,1000,10000,False)

#Perceptron without margin
p_s=Perceptron(train_x.shape[1],None)
for i in range(0,20):
    p_s.train(train_x,train_y)
print("perceptron without margin learning rate:" + str(p_s.learning_rate) + "accuracy:" + str(p_s.test(test_x,test_y)))

#Perceptron with margin 1
p_m=Perceptron_Margin(train_x.shape[1],1,None)
p_m.tune_parameter_perceptron(train_x,train_y)
for i in range(0,20):
    p_m.train(train_x,train_y)
print("perceptron margin learning rate:" + str(p_m.learning_rate) + "accuracy:" + str(p_m.test(test_x,test_y)))

#Winnow without margin
w_s = Winnow(train_x.shape[1],None)
w_s.tune_parameter_winnow(train_x,train_y)
for i in range(0,20):
    w_s.train(train_x,train_y)
print("winnow without margin alpha:" + str(w_s.alpha) + "accuracy:" + str(w_s.test(test_x,test_y)))

#Winnow with margin
w_m=Winnow_Margin(train_x.shape[1],2.0,None)
w_m.tune_parameter_winnow_margin(train_x,train_y)
for i in range(0,20):
    w_m.train(train_x,train_y)
print("winnow with margin alpha:" + str(w_m.alpha) + "margin:" + str(w_m.margin) + "accuracy:" + str(w_m.test(test_x,test_y)))

#AdaGrad
adaGrad=AdaGrad(train_x.shape[1],None)
adaGrad.tune_parameter_adagrad(train_x,train_y)
for i in range(0,20):
    adaGrad.train(train_x,train_y)
print("adagrad learning rate:" + str(adaGrad.learning_rate) + "accuracy:" + str(adaGrad.test(test_x,test_y)))








#m = 500
(train_y,train_x)=gen(10,500,1000,50000,True)
(test_y,test_x)=gen(10,500,1000,10000,False)

#Perceptron without margin
p_s=Perceptron(train_x.shape[1],None)
for i in range(0,20):
    p_s.train(train_x,train_y)
print("perceptron without margin learning rate:" + str(p_s.learning_rate) + "accuracy:" + str(p_s.test(test_x,test_y)))

#Perceptron with margin 1
p_m=Perceptron_Margin(train_x.shape[1],1,None)
p_m.tune_parameter_perceptron(train_x,train_y)
for i in range(0,20):
    p_m.train(train_x,train_y)
print("perceptron margin learning rate:" + str(p_m.learning_rate) + "accuracy:" + str(p_m.test(test_x,test_y)))

#Winnow without margin
w_s = Winnow(train_x.shape[1],None)
w_s.tune_parameter_winnow(train_x,train_y)
for i in range(0,20):
    w_s.train(train_x,train_y)
print("winnow without margin alpha:" + str(w_s.alpha) + "accuracy:" + str(w_s.test(test_x,test_y)))

#Winnow with margin
w_m=Winnow_Margin(train_x.shape[1],2.0,None)
w_m.tune_parameter_winnow_margin(train_x,train_y)
for i in range(0,20):
    w_m.train(train_x,train_y)
print("winnow with margin alpha:" + str(w_m.alpha) + "margin:" + str(w_m.margin) + "accuracy:" + str(w_m.test(test_x,test_y)))

#AdaGrad
adaGrad=AdaGrad(train_x.shape[1],None)
adaGrad.tune_parameter_adagrad(train_x,train_y)
for i in range(0,20):
    adaGrad.train(train_x,train_y)
print("adagrad learning rate:" + str(adaGrad.learning_rate) + "accuracy:" + str(adaGrad.test(test_x,test_y)))








#m = 1000
(train_y,train_x)=gen(10,1000,1000,50000,True)
(test_y,test_x)=gen(10,1000,1000,10000,False)

#Perceptron without margin
p_s=Perceptron(train_x.shape[1],None)
for i in range(0,20):
    p_s.train(train_x,train_y)
print("perceptron without margin learning rate:" + str(p_s.learning_rate) + "accuracy:" + str(p_s.test(test_x,test_y)))

#Perceptron with margin 1
p_m=Perceptron_Margin(train_x.shape[1],1,None)
p_m.tune_parameter_perceptron(train_x,train_y)
for i in range(0,20):
    p_m.train(train_x,train_y)
print("perceptron margin learning rate:" + str(p_m.learning_rate) + "accuracy:" + str(p_m.test(test_x,test_y)))

#Winnow without margin
w_s = Winnow(train_x.shape[1],None)
w_s.tune_parameter_winnow(train_x,train_y)
for i in range(0,20):
    w_s.train(train_x,train_y)
print("winnow without margin alpha:" + str(w_s.alpha) + "accuracy:" + str(w_s.test(test_x,test_y)))

#Winnow with margin
w_m=Winnow_Margin(train_x.shape[1],2.0,None)
w_m.tune_parameter_winnow_margin(train_x,train_y)
for i in range(0,20):
    w_m.train(train_x,train_y)
print("winnow with margin alpha:" + str(w_m.alpha) + "margin:" + str(w_m.margin) + "accuracy:" + str(w_m.test(test_x,test_y)))

#AdaGrad
adaGrad=AdaGrad(train_x.shape[1],None)
adaGrad.tune_parameter_adagrad(train_x,train_y)
for i in range(0,20):
    adaGrad.train(train_x,train_y)
print("adagrad learning rate:" + str(adaGrad.learning_rate) + "accuracy:" + str(adaGrad.test(test_x,test_y)))




