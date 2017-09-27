import numpy as np


""" 
Funtion to implement Perceptron without margin

-----------
Parameters:

size: size of different data set

learning_rate: take in learning_rate with or without parameter tuning

--------------
Example usage:

train data

test hypothesis

-----
"""


class Perceptron:
    def __init__(self, size, learning_rate):
        self.bias = 0  
        self.margin = 0
        self.new_mistake = 0
        self.total =[0]
        self.mistake = [0]
        self.weight = np.zeros(size)
        self.learning_rate = learning_rate if learning_rate != None else 1
    def train(self, x_vector, y_vector):
        for i in range(0,len(x_vector)):
            self.total.append(i+1)
            temp = self.calculate(x_vector[i], y_vector[i])
            if  temp <= self.margin:
                self.update(x_vector[i], y_vector[i])
            if temp <= 0:
                self.mistake.append(self.mistake[i]+1)
            else:
                self.mistake.append(self.mistake[i])
        pass
    def test(self, x_vector, y_vector):
        correct_count = 0
        for i in range(0,len(x_vector)):
            if self.calculate(x_vector[i], y_vector[i]) > 0:
                correct_count = correct_count + 1
        return correct_count/len(x_vector)
    def update(self, temp_x, temp_y):
        self.weight = self.weight + self.learning_rate*temp_y*temp_x
        self.bias = self.bias + self.learning_rate*temp_y
        pass
    def calculate(self, temp_x, temp_y):
        temp = sum(self.weight * temp_x)
        temp = temp + self.bias
        temp = temp * temp_y
        return temp
    def converge(self, x_vector, y_vector):
        i = 0
        while i < 1000:
            for j in range(0,len(x_vector)):
                temp = y_vector[j]*(sum(self.weight*x_vector[j]) + self.bias)
                if temp <= self.margin:
                    self.update(x_vector[j],y_vector[j])
                if temp <= 0:
                    self.new_mistake=self.new_mistake+1
                    i = 0
                else:
                    i = i + 1
        pass


