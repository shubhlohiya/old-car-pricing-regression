import numpy as np
import matplotlib.pyplot as plt
import math
import random

class Neural_Network():
    def __init__(self, neurons, Activations, initialization='randn'): 
        # arguments: an array "neurons" consist of number of neurons for each layer, 
        # an array "activations" consisting of activation functions used for the hidden layers and output layer
        self.inputSize = neurons[0] # Number of neurons in input layer
        self.outputSize = neurons[-1] # Number of neurons in output layer
        self.layers = len(neurons)
        self.weights = [] #weights for each layer
        self.biases = [] #biases in each layer 
        self.layer_activations = [] #activations in each layer
        if initialization=='rand':
            self.initializer=np.random.rand
        elif initialization=='randn':
            self.initializer=np.random.randn
        for i in range(len(neurons)-1): 
            # changed from rand to randn here as those give better training results
            self.weights.append(self.initializer(neurons[i+1],neurons[i])) #weight matrix between layer i and layer i+1
            self.biases.append(self.initializer(neurons[i+1],1))
            self.layer_activations.append(Activations[i]) #activations for each layer
        
            
    def sigmoid(self, z): # sigmoid activation function
        #Fill in the details to compute and return the sigmoid activation function                  
        return 1.0/(1.0+np.exp(-z))
    
    def sigmoidPrime(self,z): # derivative of sigmoid activation function
        #Fill in the details to compute and return the derivative of sigmoid activation function
        return self.sigmoid(z)*(1-self.sigmoid(z))

                          
    def tanh(self, z): # hyperbolic tan activation function
        #Fill in the details to compute and return the tanh activation function                  
        a = np.exp(z)
        return (a-1/a)/(a+1/a)
    
    def tanhPrime(self,x): # derivative of hyperbolic tan activation function
        #Fill in the details to compute and return the derivative of tanh activation function
        return 1-self.tanh(x)**2
                          
    def linear(self, z): # Linear activation function
        #Fill in the details to compute and return the linear activation function                                    
        return z
    
    def linearPrime(self,z): # derivative of linear activation function
        #Fill in the details to compute and return the derivative of activation function                                                      
        return np.ones(z.shape)

    def ReLU(self,z): # ReLU activation function
        #Fill in the details to compute and return the ReLU activation function                  
        return np.maximum(0,z)
    
    def ReLUPrime(self,z): # derivative of ReLU activation function
        #Fill in the details to compute and return the derivative of ReLU activation function
        return 1 * (z>0)
    
    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z), axis = 0)
    
    def forward(self, a): # function of forward pass which will receive input and give the output of final layer
        # Write the forward pass using the weights and biases to find the predicted value and return them.
        layer_activations_a = [a] #store the input as the input layer activations
        layer_dot_prod_z = []
        for i, param in enumerate(zip(self.biases, self.weights)):
            b, w = param[0], param[1]
            z = np.dot(w, a)+b
            if self.layer_activations[i].lower()=='sigmoid':
#                 z = np.dot(w, a)+b
                a = self.sigmoid(z)
            elif self.layer_activations[i].lower()=='relu':
                a = self.ReLU(z)
            elif self.layer_activations[i].lower()=='tanh':   
                a = self.tanh(z)
            elif self.layer_activations[i].lower()=='linear':
                a = self.linear(z)
            elif self.layer_activations[i].lower()=='softmax':
                a = self.softmax(z)
            layer_dot_prod_z.append(z)    
            layer_activations_a.append(a)
        return a, layer_dot_prod_z, layer_activations_a
                          
            
    
    def backward(self, x, y, zs, activations): # find the loss and return derivative of loss w.r.t every parameter
        # Write the backpropagation algorithm here to find the gradients of weights and biases and return them.
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]
        
        m = zs[-1].shape[1] #if len(zs[-1].shape)==2 else 1
        
        # backward pass
        if self.layer_activations[-1].lower()=='sigmoid':
            delta = (activations[-1] - y) * \
                self.sigmoidPrime(zs[-1])
        elif self.layer_activations[-1].lower()=='relu':
            delta = (activations[-1] - y) * \
                self.ReLUPrime(zs[-1])
        elif self.layer_activations[-1].lower()=='tanh':   
            delta = (activations[-1] - y) * \
                self.tanhPrime(zs[-1])
        elif self.layer_activations[-1].lower()=='linear':
            delta = (activations[-1] - y) * \
                self.linearPrime(zs[-1])
        elif self.layer_activations[-1].lower()=='softmax':
            delta = activations[-1] - y

        # fill in the appropriate details for gradients of w and b
        grad_b[-1] = np.sum(delta, axis=1, keepdims=True)/m
        grad_w[-1] = np.dot(delta, activations[-2].T)/m
                 
        for l in range(2, self.layers): # Here l is in backward sense i.e. last l th layer
            z = zs[-l]
            if self.layer_activations[-l].lower()=='sigmoid':
                prime = self.sigmoidPrime(z)
            elif self.layer_activations[-l].lower()=='relu':
                prime = self.ReLUPrime(z)
            elif self.layer_activations[-l].lower()=='tanh':   
                prime = self.tanhPrime(z)
            elif self.layer_activations[-l].lower()=='linear':
                prime = self.linearPrime(z)

            #Compute delta, gradients of b and w
            delta = prime * np.dot(self.weights[-l+1].T, delta) # delta is dz
            grad_b[-l] = np.sum(delta, axis=1, keepdims=True)/m
            grad_w[-l] = np.dot(delta, activations[-l-1].T)/m
                          
        return (grad_b, grad_w)                 

    def update_parameters(self, grads, learning_rate): # update the parameters using the gradients
        # update weights and biases using the gradients and the learning rate
        
        grad_b, grad_w = grads[0], grads[1]       
        
        #Implement the update rule for weights  and biases
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate*grad_w[i]
            self.biases[i] -= learning_rate*grad_b[i]

    def RMSE(self, predicted, actual) :
        # Root Mean Squared Error
        return np.sqrt(np.sum((1e4*(predicted - actual)) ** 2) / len(actual))
    
    def live_error(self, X_train, Y_train, Xval, Yval, S1_errors, S2_errors):
        y1 = np.squeeze(self.forward(X_train)[0])
        S1_errors.append(self.RMSE(y1, Y_train))
        y2 = np.squeeze(self.forward(Xval)[0])
        S2_errors.append(self.RMSE(y2, Yval))
        pass
                     
    def train(self, X, Y, lr=1e-6, p=5, cap=1000, bs=20, Xval=None, Yval=None):
            
        S1_errors=[]
        S2_errors=[]

        i,j,v=0,0,np.inf
    
        while j<p:
            for q in range(0,len(X[0]),bs):
                start, end = q*bs, min(len(X[0]), (q+1)*bs)
                train_x = X[:, start:end] 
                train_y = Y[start:end]
                out, dot_prod_z, activations_a = self.forward(train_x)
                grads = self.backward(train_x, train_y, dot_prod_z, activations_a) # find the gradients using backward pass
                self.update_parameters(grads, lr)
            i+=1

            self.live_error(X, Y, Xval, Yval, S1_errors, S2_errors)
            v_new = S2_errors[-1]
            if v_new < v:
                j=0
                v = v_new
            else:
                j+=1
            print(f"Epoch {i}..............RMSE on train = {S1_errors[-1]}, RMSE on dev = {S2_errors[-1]}")

            if i >= cap:
                break
        if i >= cap:
            print("Reached Epoch Cap without convergence....Terminating")
        else:
            print("Early Stopping .............. Returning best weights")
        
        x = [(i+1) for i in range(len(S1_errors))]
        plt.plot(x, S1_errors, label="Loss on Train")
        plt.plot(x, S2_errors, label="Loss on Eval")
        plt.legend()
        plt.title(f"Learning Rate = {lr}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()
        
    # def predict(self, x):
    #     print ("Input : \n" + str(x))
    #     prediction,_,_ = self.forward(x)
    #     print ("Output: \n" + str(prediction))