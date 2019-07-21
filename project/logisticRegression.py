#-*- coding:utf-8 -*-
import numpy as np

def sigmoid(z):
	s = 1.0/(1.0+np.exp(-z))
	return s
def costFunc(params, X, y, reg):
	'''
	cost function
	X shape is [M,N], N is features number, M is samples number
	y shape is [M,1]
	params shape is [N,1]
	return cost value
	'''
	m = len(y)#num of sample
	z = np.dot(X, params)#shape[M,1]
	h = sigmoid(z)#[M,1]
	'''
	print('X',X)
	print('z',z)
	print('h',h)
	#[1,M]*[M,1] = [1,1]
	print('y.T',y.T)
	print('np.log(h)',np.log(h))
	print('-y.T*np.log(h):',-np.dot(y.T,np.log(h)))
	print('-(1-y).T*np.log(1-h):',np.dot((1-y).T,np.log(1-h)))
	'''
	#J = (-np.dot(y.T,np.log(h))-np.dot((1-y).T,np.log(1-h)))/m
	#add reg
	J = (-np.dot(y.T,np.log(h))-np.dot((1-y).T,np.log(1-h)))/m + (reg/(2*m))*np.sum(np.dot(params.T,params))
	return J
def gradient(params, X, y, reg):
	'''
	calculate gradient of cost function
	'''	
	h=sigmoid(np.dot(X, params))#[M,1]
	M = X.shape[0]
	#grad = np.dot(X.T, (h-y))
	#add reg
	grad = np.dot(X.T, (h-y)) + (reg/M)*params #[N,M]*[M,1]=[N,1]
	return grad
def logistic_gradientDescent(params, X, y, alpha, reg):
	'''
	Gradient descent
	'''

	grad = gradient(params, X, y, reg)#[N,1]
	J = costFunc(params, X, y, reg)
	#params[N,1] 
	params = params - alpha*grad
	return params, J

def classify(params, X):
	'''
	classify
	'''
	z = np.dot(X, params)#shape[M,1]
	h = sigmoid(z)#[M,1]
	# y = np.zeros((X.shape[0],1))
	# for i in range(h.shape[0]):
	# 	y[i] = 1 if h[0] >= 0.5 else 0
	y = np.round(h)#easy method
	return y#[M,1]

def pricise(y, y_predict):
	'''
	return pricise of predict
	'''
	same = 0.0#cution! if there is a int,pricise always 0
	M = y.shape[0]
	for i in range(M):
		#print('y_predict['+str(i)+']',y_predict[i])
		#print('y['+str(i)+']:',y[i])
		if y_predict[i] == y[i]:
			same += 1
	pricise = same/M
	#print pricise
	return pricise