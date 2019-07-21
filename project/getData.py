#-*- coding:utf-8 -*-
import numpy as np
import glob
import cv2

def get_data(path,train_flag=False):
	'''
	get origin data
	1. RESIZE images to same size 224*224
	2. and transfer rgb to GRAY
	return X and y in np.array
	X[M,N] y[M,1]
	'''
	normal_dir = path+'NORMAL/'
	pneumonia_dir = path+'PNEUMONIA/'

	normal_cases = glob.glob(normal_dir+'*.jpeg')
	pneumonia_cases = glob.glob(pneumonia_dir+'*.jpeg')

	data_list = []
	for img in normal_cases:
		data_list.append((img,0))
	# print(train_data[0])
	for img in pneumonia_cases:
		data_list.append((img,1))

	#X
	X_list = []
	y_list = []
	print 'reading '+path+' data...'
	flag = -1
	num_train = 624
	for data in data_list:
		flag += 1
		if train_flag:
			if flag>(num_train/2)+1 and flag<len(data_list)-(num_train/2):
				continue
		img = cv2.imread(data[0])
		img = cv2.resize(img,(224,224))#resize all images to same size
		gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#rgb to gray
		temp = np.array(gray_img)
		temp = temp.reshape(1,-1)
		#print('temp shape',temp.shape)
		X_list.append(temp)
		y_list.append(data[1])
		#save test time

	y = np.array(y_list)
	y = y.reshape(-1,1)#[M,1]
	X = np.array(X_list)
	X = X.reshape(y.shape[0],-1)#[M,N]

	return X, y

def data_prepro(X, n=1):
	'''
	preprocessing data
	n=1: normalization the pixel values by devide 255
	n=2: normalization by Z-Score (x-miu)/sigma
	'''
	if n==1:#归一化
		print('normalization(/255)...')
		X1 = X/255
		return X1
	if n==2:#0均值化再除以标准差
		print('normalization((x-miu)/sigma)...')
		means = np.mean(X,axis=1)#means of every row
		means = means.reshape(-1,1)#[M,1]
		std = np.std(X,axis=1)
		std = std.reshape(-1,1)#[M,1]
		X2 = (X - means)/std
		return X2
