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
Function that test the algoritem and plot for question 2

______________

"""







y_vector = []
x_vector = []
total = []
mistake = []
temp_mistake = []
for i in range(0,5):
    (y, x) = gen(10, 20, 40*(i+1), 50000, False)
    y_vector.append(y)
    x_vector.append(x)
    total.append((i+1)*40)
winnow_m=[]
adaGrad=[]


#Simple Perceptron
p_s = []
for i in range(0,5):
    size=x_vector[i].shape[1]
    p_s.append(Perceptron(size,None))
    print("n=" + str((i+1)*40) + "perceptron without margin learning rate:" + str(p_s[i].learning_rate))
    p_s[i].converge(x_vector[i],y_vector[i])
    temp_mistake.append(p_s[i].new_mistake)
mistake.append(temp_mistake)


#Perceptron with margin
temp_mistake = []
p_m = []
for i in range(0,5):
    size=x_vector[i].shape[1]
    p_m.append(Perceptron_Margin(size,1,None))
    p_m[i].tune_parameter_perceptron(x_vector[i],y_vector[i])
    print("n=" + str((i+1)*40) + "perceptron with margin learning rate:" + str(p_m[i].learning_rate))
    p_m[i].converge(x_vector[i],y_vector[i])
    temp_mistake.append(p_m[i].new_mistake)
mistake.append(temp_mistake)


#Winnow without margin
temp_mistake = []
w_s = []
for i in range(0,5):
    size=x_vector[i].shape[1]
    w_s.append(Winnow(size,None))
    w_s[i].tune_parameter_winnow(x_vector[i],y_vector[i])
    print("n=" + str((i+1)*40) + "winnow without margin alpha:" + str(w_s[i].alpha))
    w_s[i].converge(x_vector[i],y_vector[i])
    temp_mistake.append(w_s[i].new_mistake)
mistake.append(temp_mistake)


#Winnow with margin
temp_mistake = []
w_m = []
for i in range(0,5):
    size=x_vector[i].shape[1]
    w_m.append(Winnow_Margin(size,2.0,None))
    w_m[i].tune_parameter_winnow_margin(x_vector[i],y_vector[i])
    print("n=" + str((i+1)*40) + "winnot with margin alpha:" + str(w_m[i].alpha) + "winnot with margin margin:" + str(w_m[i].margin))
    w_m[i].converge(x_vector[i],y_vector[i])
    temp_mistake.append(w_m[i].new_mistake)
mistake.append(temp_mistake)


#AdaGrad
temp_mistake = []
adaGrad = []
for i in range(0,5):
    size=x_vector[i].shape[1]
    adaGrad.append(AdaGrad(size,None))
    adaGrad[i].tune_parameter_adagrad(x_vector[i],y_vector[i])
    print("n=" + str((i+1)*40) + "adagrad learning rate:" + str(adaGrad[i].learning_rate))
    adaGrad[i].converge(x_vector[i],y_vector[i])
    temp_mistake.append(adaGrad[i].new_mistake)
mistake.append(temp_mistake)


for i in range(0,5):
    plt.plot(total,mistake[i])
plt.legend(['perceptron', 'perceptron with margin', 'winnow', 'winnow with margin',"adaGrad"], loc='upper left')
plt.show()


