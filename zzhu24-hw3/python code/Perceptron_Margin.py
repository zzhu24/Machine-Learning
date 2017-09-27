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


class Perceptron_Margin:
    def __init__(self, size, margin, learning_rate):
        self.bias= 0  
        self.new_mistake=0
        self.total = [0]
        self.mistake = [0]
        self.weight=np.zeros(size)
        self.margin=margin if margin > 0 else 0
        self.learning_rate= learning_rate if learning_rate != None else 1
    def tune_parameter_perceptron(self, x_vector, y_vector):
        i = 0
        while i < 1000:
            for j in range(0,len(x_vector)):
                temp = self.calculate(x_vector[j], y_vector[j])
                if  temp < self.margin:
                    self.update(x_vector[j], y_vector[j])
                if temp <= 0:
                    self.new_mistake= self.new_mistake+1
                    i = 0
                else:
                    i = i+1
        pass
    def train(self, x_vector, y_vector):
        for i in range(0,len(x_vector)):
            self.total.append(i+1)
            temp = self.calculate(x_vector[i], y_vector[i])
            if  temp < self.margin:
                self.update(x_vector[i],y_vector[i])
            if temp<=0:
                self.mistake.append(self.mistake[i]+1)
            else:
                self.mistake.append(self.mistake[i])
        pass
    def update(self, temp_x, temp_y):
        self.weight = self.weight + self.learning_rate * temp_y * temp_x
        self.bias = self.bias + self.learning_rate * temp_y
        pass
    def test(self, x_vector, y_vector):
        correct_count = 0
        for i in range(0,len(x_vector)):
            if self.calculate(x_vector[i], y_vector[i]) > 0:
                correct_count = correct_count + 1
        return correct_count/len(x_vector)
    def calculate(self, temp_x, temp_y):
        temp = sum(self.weight * temp_x)
        temp = temp + self.bias
        temp = temp * temp_y
        return temp
    def tune_parameter_perceptron(self, x_vector, y_vector):
        D1_x = self.subset(x_vector)
        D1_y = self.subset(y_vector)
        D2_x = self.subset(x_vector)
        D2_y = self.subset(y_vector)
        accuracy=[]
        choose_from = [1.5, 0.25, 0.03, 0.005, 0.001]
        for i in choose_from:
            self.learning_rate = i
            self.train(D1_x,D1_y)
            self.total=[0]
            self.mistake=[0]
            right = self.test(D2_x,D2_y)
            accuracy.append(right)
            self.bias = 0
            self.weight=np.zeros(self.weight.size)
        self.learning_rate = choose_from[np.argmax(accuracy)]
        pass
    def subset(self, data):
        temp = []
        for i in range(0, int(len(data) * 0.1)):
            temp.append(data[np.random.randint(len(data))])
        return np.array(temp)
    def converge(self, x_vector, y_vector):
        i = 0
        while i<1000:
            for j in range(0,len(x_vector)):
                temp = self.calculate(x_vector[j], y_vector[j])
                if temp < self.margin:
                    self.update(x_vector[j], y_vector[j])
                if temp <= 0:
                    self.new_mistake = self.new_mistake+1
                    i = 0
                else:
                    i = i + 1
        pass
            