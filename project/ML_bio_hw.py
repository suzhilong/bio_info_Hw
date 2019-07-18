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

def data_prepro():
	'''
	preprocessing data
	'''
	pass

def train(data, n=1):
	'''
	trainning model
	There are three classfy mathods
	n=1, logistic
	n=2, SVM
	n=3, neural network
	Save and return parameters of model
	'''
	pass

def classify(params, data):
	'''
	classify
	'''
	pass

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