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




class AdaGrad:
    def __init__(self,size,learning_rate):
        self.bias = 0  
        self.hingeLoss= 0
        self.new_mistake=0
        self.total = [0]
        self.mistake = [0]
        self.weight = np.zeros(size)
        self.gradient = np.zeros(size+1)
        self.learning_rate= learning_rate if learning_rate != None else 1.5
    def train(self, x_vector, y_vector):
        for i in range(0,len(x_vector)):
            self.total.append(i+1)
            temp = self.calculate(x_vector[i], y_vector[i])
            if temp <= 1:
                g_t = np.append(-y_vector[i]*x_vector[i],-y_vector[i])
                self.gradient = self.gradient + g_t**2
                self.update(x_vector[i],y_vector[i])
            self.hingeLoss= self.hingeLoss + max(0,1-temp)
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
        self.weight = self.weight + self.vector_division(self.learning_rate*temp_y*temp_x,np.sqrt(self.gradient[0 : self.gradient.size-1]))
        self.bias = self.bias + self.learning_rate*temp_y/np.sqrt(self.gradient[self.gradient.size-1])
        #self.weight = self.weight + self.learning_rate * temp_y * temp_x
        #self.bias = self.bias + self.learning_rate * temp_y
        pass
    def test(self, x_vector, y_vector):
        correct_count = 0    
        for i in range(0,len(x_vector)):
            temp = self.calculate(x_vector[i], y_vector[i])
            self.hingeLoss= self.hingeLoss + max(0, 1-temp)
            if temp  > 0:
                correct_count = correct_count + 1
        return correct_count/len(x_vector)
    def vector_division(self,a,b):
        if a.size!=b.size:
            return None
        for i in range(0,a.size):
            if b[i] != 0:
                a[i]= a[i]/b[i]
        return a
    def tune_parameter_adagrad(self, x_vector, y_vector):
        D1_x = self.subset(x_vector)
        D1_y = self.subset(y_vector)
        D2_x = self.subset(x_vector)
        D2_y = self.subset(y_vector)
        accuracy=[]
        learning_rate=[1.5, 0.25, 0.03, 0.005, 0.001]
        for i in learning_rate:
            self.learning_rate = i
            self.train(D1_x,D1_y)
            self.mistake = [0]
            self.total = [0]
            accuracy.append(self.test(D2_x,D2_y))
            self.bias = 0
            self.weight = np.zeros(self.weight.size)
            self.gradient = np.zeros(self.gradient.size)
        self.learning_rate = learning_rate[np.argmax(accuracy)]
        
        #print(accuracy)
        pass
    def subset(self, data):
        temp = []
        for i in range(0, int(len(data) * 0.1)):
            temp.append(data[np.random.randint(len(data))])
        return np.array(temp)
    def converge(self, x_vector, y_vector):
        R=0
        while R<1000:
             for i in range(0,len(x_vector)):
                temp = self.calculate(x_vector[i], y_vector[i])
                if temp <= 1:
                    g_t = np.append(-y_vector[i]*x_vector[i],-y_vector[i])
                    self.gradient = self.gradient + g_t**2
                    self.update(x_vector[i],y_vector[i])
                if temp <= 0:
                    self.new_mistake=self.new_mistake+1
                    R=0
                else:
                    R=R+1
        pass