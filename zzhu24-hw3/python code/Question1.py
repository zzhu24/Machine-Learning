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
Function that test the algoritem and plot for question 1

______________

change the first line to access different n
"""

(y, x) = gen(10, 100, 500, 50000, False)
#(y, x) = gen(10, 100, 1000, 50000, False)

#Simple Perceptron
p_s=Perceptron(x.shape[1], None)
p_s.train(x,y)
print("learning rate of Simple Perceptron     " +  str(p_s.learning_rate))

#Perceptron with margin
p_m=Perceptron_Margin(x.shape[1], 1, None)
p_m.tune_parameter_perceptron(x,y)
p_m.train(x,y)
print("learning rate of Margined Perceptron     " + str(p_m.learning_rate))
""
#Winnow without margin
w_s=Winnow(x.shape[1], None)
w_s.tune_parameter_winnow(x,y)
w_s.train(x,y)
print("alpha of Simple Winnow     " +  str(w_s.alpha))

#Winnow with margin
w_m = Winnow_Margin(x.shape[1],2.0,None)
w_m.tune_parameter_winnow_margin(x,y)
w_m.train(x,y)
print("alpha of Simple Winnow     " +  str(w_m.alpha))
print("margin of Simple Winnow     " +  str(w_m.margin))



a1=AdaGrad(x.shape[1],None)
a1.tune_parameter_adagrad(x,y)
a1.train(x,y)
print("learning rate of adagrad     " + str(a1.learning_rate))

plt.plot(p_s.total, p_s.mistake)
plt.plot(p_m.total,p_m.mistake)
plt.plot(w_s.total,w_s.mistake)
plt.plot(w_m.total,w_m.mistake)
plt.plot(a1.total,a1.mistake)
plt.legend(['perceptron', 'perceptron with margin', 'winnow', "winnow with margin", "adagrad"], loc='upper right')
plt.show()

