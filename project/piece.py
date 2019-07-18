import numpy as np
import matplotlib.pyplot as plt

#logistic

def sigmoid(z):
	s = 1/(1+np.exp(-z))
	return s

def costFun(params, X, y):
	'''
	cost function
	X shape is [M,N], N is features number, M is samples number
	y shape is [M,1]
	params shape is [N,1]
	return cost value
	'''
	z = np.dot(X, params)#shape[M,1]
	h = sigmoid(z)#[M,1]
	#[1,M]*[M,1] = [1,1]
	J = (-np.dot(np.transpose(y),np.log(h))-np.dot(np.transpose(1-y),np.log(1-h)))/len(y)
	return J

def gradient(params, X, y):
	'''
	calculate gradient of cost function
	'''	
	h=sigmoid(np.dot(X, params))#[M,1]
	grad = np.dot(X.T, (y-h))#[N,M]*[M,1]=[N,1]
	return grad

def gradientDescent(params, X, y, alpha, iterations):
	'''
	Gradient descent
	'''
	J_trace = np.zeros((iterations, 1))
	for i in range(iterations):
		grad = gradient(params, X, y)
		params = params - alpha*grad
		J_trace[i] = costFun(params, X, y)

	return params, J_trace

def main():
	a=1 if 1==1 else 0
	print a
if __name__ == '__main__':
	main()

