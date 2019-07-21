#python2
#-*- coding:utf-8 -*-
#Hw of bio_info
#created by suzhilong
#2019.7

import numpy as np
import matplotlib.pyplot as plt
from getData import get_data,data_prepro
from logisticRegression import *
from SVM import *
from neuralNetwork import NN


def train(init_params, X_train, X_test, y_train, y_test, alpha, reg, iterations, n=1):
	'''
	trainning model
	There are three classfy methods
	n=1, logistic
	n=2, SVM
	n=3, neural network
	Save and return parameters of model and every iteration J
	'''
	params = init_params
	if n == 1:#logistic regression
		J_train = np.zeros((iterations, 1),dtype='float64')
		J_test = np.zeros((iterations, 1),dtype='float64')
		pricises = np.zeros((iterations, 1),dtype='float64')
		i = 0
		while i<iterations:
			params, J_train[i] = logistic_gradientDescent(params, X_train, y_train, alpha, reg)
			J_test[i] = costFunc(params, X_test, y_test, reg)
			y_predict = classify(params, X_test)
			pricises[i] = pricise(y_test, y_predict)
			print 'iteration'+str(i)+':','alpha='+str(alpha),'reg='+str(reg) ,'J_train='+str(J_train[i][0]),'J_test='+str(J_test[i][0]),'pricise='+str(pricises[i][0])
			if abs(pricises[i]-pricises[i-1])<0.00001:
				#break
				pass
			i += 1
		return params, J_train, J_test, pricises #every iteration of function J
	if n == 2:#SVM
		model = SVM(X_train,y_train)
		#print model.predict(X_train)
		#print model.predict(X_test)
		acc_train,acc_test = evalue(model, X_train,y_train,X_test,y_test)
		return acc_train,acc_test
	if n == 3:#nn
		X_train = X_train.T
		y_train = y_train.reshape(y_train.shape[0], -1).T
		X_test = X_test.T
		y_test = y_test.reshape(y_test.shape[0], -1).T
		layer_dims = [X_train.shape[0],10,5,1]
		alpha = 0.1
		iterations = 1000
		accurate = NN(X_train,y_train,X_test,y_test,layer_dims,alpha,iterations)
		return accurate

def main():
#get_data
	path_train = '/home/su/code/bio_Hw/chest_xray/chest_xray/train/'
	path_test = '/home/su/code/bio_Hw/chest_xray/chest_xray/test/'
	#path_val = ''
	X_train_origin,y_train = get_data(path_train,True)
	X_test_origin,y_test = get_data(path_test)
	#print 'shape of X_train_origin:', X_train_origin.shape
	#print 'shape of y_train:', y_train.shape
	#print 'shape of X_test_origin:', X_test_origin.shape
	#print 'shape of y_test:', y_test.shape 

#data preprocessing
	#1:devide 255 to normalization 2: minas means then devide standarded error
	n_pre = 1
	X_train =data_prepro(X_train_origin,n_pre)
	X_test = data_prepro(X_test_origin,n_pre)
	#print 'shape of X_train:',X_train.shape
	#print 'shape of X_test:',X_test.shape

#1:logistic 2:SVM 3:neural network
	n_train = 3
#train
	alpha =0.01
	reg = 1000
	iterations = 500
	init_params = np.random.standard_normal((X_train.shape[1],1))#[N,1]
	if n_train==1:#logistic
		params, J_train, J_test, pricises = train(init_params, X_train, X_test, y_train, y_test, alpha, reg, iterations, n_train)
		#print('params:',params)
		#print('J:',J)
		
		#classification
		y_predict = classify(params, X_test)
		#print('y_predict',y_predict)
		pricise_final = pricise(y_test, y_predict)
		print 'The final pricise of predict is:',pricise_final
		
		#plot
		fig_J_train, = plt.plot(J_train)
		fig_J_test, = plt.plot(J_test)
		plt.legend(handles=[fig_J_train, fig_J_test], labels=['J_train', 'J_test'], loc='upper right')
		plt.title('num of samples='+str(y_predict.shape[0])+'   '+'alpha='+str(alpha)+'   '+'reg='+str(reg))
		plt.show()
		plt.plot(pricises)
		plt.title('num of samples='+str(y_predict.shape[0])+'   '+'alpha='+str(alpha)+'   '+'reg='+str(reg))
		plt.show()
	elif n_train==2:#SVM
		acc_train,acc_test = train(init_params, X_train, X_test, y_train, y_test, alpha, reg, iterations, n_train)
		print "accurate of train：" + str(acc_train)
		print "accurate of test:" + str(acc_test)
	elif n_train==3:#nn
		accurate = train(init_params, X_train, X_test, y_train, y_test, alpha, reg, iterations, n_train)
		print 'accurate of test:',accurate

if __name__ == '__main__':
	main()