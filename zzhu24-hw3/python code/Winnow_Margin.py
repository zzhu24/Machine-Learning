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




class Winnow_Margin:
    def __init__(self,size,margin,alpha):
        self.weight=np.ones(size)
        self.bias= -size 
        self.margin=margin if margin > 0 else 0
        self.alpha= alpha if alpha != None else 1.1
        self.mistake = [0]
        self.total =[0]
        self.new_mistake=0
    def train(self, x_vector, y_vector):
        for i in range(0,len(x_vector)):
            self.total.append(i+1)
            temp = self.calculate(x_vector[i], y_vector[i])
            if temp < self.margin:
                self.update(x_vector[i],y_vector[i])
            if temp <= 0:
                self.mistake.append(self.mistake[i]+1)
            else:
                self.mistake.append(self.mistake[i])
        pass
    def calculate(self, temp_x, temp_y):
        temp = sum(self.weight * temp_x)
        temp = temp + self.bias
        temp = temp * temp_y
        return temp
    def update(self, temp_x, temp_y):
        self.weight = self.weight * self.alpha ** (temp_y * temp_x)
        #self.weight = self.weight + self.learning_rate * temp_y * temp_x
        #self.bias = self.bias + self.learning_rate * temp_y
        pass
    def test(self, x_vector, y_vector):
        correct_count = 0
        for i in range(0,len(x_vector)):
            if self.calculate(x_vector[i], y_vector[i])> 0:
                correct_count = correct_count + 1 
        return correct_count/len(x_vector)
    def tune_parameter_winnow_margin(self, x_vector, y_vector):
        D1_x = self.subset(x_vector)
        D1_y = self.subset(y_vector)
        D2_x = self.subset(x_vector)
        D2_y = self.subset(y_vector)
        accuracy=[]
        alpha =[1.1, 1.01, 1.005, 1.0005, 1.0001]
        margin = [0.3, 2.0, 0.04, 0.006, 0.001]
        for i in alpha:
            self.alpha=i
            for j in margin:
                self.margin = j
                self.train(D1_x,D1_y)
                self.mistake = [0]
                self.total = [0]
                accuracy.append(self.test(D2_x,D2_y))
                self.bias = -self.weight.size
                self.weight=np.ones(self.weight.size)
        self.alpha = alpha[np.argmax(accuracy)//len(margin)]
        self.margin = margin[np.argmax(accuracy)%len(alpha)]
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