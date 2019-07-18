#python3
#-*- coding:utf-8 -*-
#Hw of bio_info
#created by suzhilong
#2019.7

import numpy as np
import matplotlib.pyplot as plt
import os


def get_data(path):
	'''
	get origin data
	return x and y
	'''
	pass

def data_prepro(data):
	'''
	preprocessing data
	'''
	pass

def train(init_params, X, y, alpha, iterations, n=1):
	'''
	trainning model
	There are three classfy mathods
	n=1, logistic
	n=2, SVM
	n=3, neural network
	Save and return parameters of model
	'''
	if n == 1:
		logistic_gradientDescent(params, X, y, alpha, iterations)

def classify(params, X):
	'''
	classify
	'''
	z = np.dot(X, params)#shape[M,1]
	h = sigmoid(z)#[M,1]
	y = np.zeros((X.shape[0],1))
	for i in range(h.shape[0]):
		y[i] = 1 if h[0] >= 0.5 else 0
	return y#[M,1]

def precise(params, X, y):
	'''
	return precise of predict
	'''
	y_predict = classify(params, X)#[M,1]
	same = 0
	M = y.shape[0]
	for i in range(M):
		if y_predict[i] == y[i]:
			same += 1
	precise = same/M
	return precise	

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
def logistic_gradientDescent(params, X, y, alpha, iterations):
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
#get_data
	path_train = ''
	path_test = ''
	path_val = ''
	x_train_origin,y_train = get_data(path_train)
	x_test_origin,x_test = get_data(path_test)
	x_val_origin,x_val = get_data(path_val)

#data preprocessing
	x_train =data_prepro()
	x_test = data_prepro()
	x_val = data_prepro()

#train
	params = train(x_train)

#classification
	y = classify(params, x_test)

if __name__ == '__main__':
	main()