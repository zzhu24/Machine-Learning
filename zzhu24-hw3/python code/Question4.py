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
Function that test the algoritem and plot for question 4
"""

loss=[]
time=[]
mistake = []

(data_y,data_x)=gen(10,20,40,10000,True)
adaGrad=AdaGrad(data_x.shape[1],None)
for i in range(0,50):
    adaGrad.hingeLoss = 0
    adaGrad.mistake=[0]
    #pring(adaGrad.mistake)
    adaGrad.time=[0]
    time.append(i+1)
    adaGrad.train(data_x,data_y)
    #print(adaGrad.hingloss)
    adaGrad.hingeLoss=0
    mistake.append((1-adaGrad.test(data_x,data_y))*len(data_x))
    loss.append(adaGrad.hingeLoss)
    adaGrad.hingeLoss = 0
    adaGrad.mistake=[0]
    adaGrad.time=[0]

plt.figure()
plt.plot(time,mistake)
plt.legend(['Mistakes'], loc='upper right')
#plt.set_ylim([900,1100])
plt.show()
plt.figure()
plt.plot(time,loss)
plt.legend(['Loss function'], loc='upper right')
#plt.set_ylim([1100,1400])
plt.show()







