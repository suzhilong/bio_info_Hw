'''
k_means
copy right:Suzhilong
2019.4.28
Hw1 of Bio_info
python version 2.7
'''

# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import numpy as np

def generateData(dataNum):
	'''
	generate data
	dataNum is the numbers of data
	Each data have 2 dimonsion like (x,y) which 
	both subject to Gaussian distribution
	return a np array like (x,y)
	'''
	mean=np.array([0,0])
	var=np.array(([2,0],[0,2]))
	#以mean为均值，var为协方差矩阵，生成正态分布的随机数
	x1=np.random.multivariate_normal(mean-1,var,dataNum/2)
	x2=np.random.multivariate_normal(mean+1,var,dataNum/2)
	#print x1,x2
	x=np.append(x1,x2,axis=0)
	#print type(x)
	return x

def getClassNum(point,center1,center2,center3):
	'''
	输入一个二维点和三个类的中心点
	计算点离那个类别最近,返回最近类别号
	'''
	dist1 = np.sqrt((point[0]-center1[0])**2 + (point[1]-center1[1])**2)
	dist2 = np.sqrt((point[0]-center2[0])**2 + (point[1]-center2[1])**2)
	dist3 = np.sqrt((point[0]-center3[0])**2 + (point[1]-center3[1])**2)
	minDist = dist1
	if minDist >= dist2:
		minDist = dist2
	if minDist >= dist3:
		minDist = dist3

	if minDist == dist1:
		return 1
	elif minDist == dist2:
		return 2
	else:
		return 3

def getCenter(cls):
	'''
	输入一个np.array列表,列表中的二维点都是同一个类
	求这些点的中心
	返回一个二维点
	'''
	sumx = 0
	sumy = 0
	#print cls
	for p in cls:
		#print p
		sumx += p[0]
		sumy += p[1]
	x = sumx / len(cls)
	y = sumy / len(cls)
	p = [x,y]
	return p

def classify(data,center1,center2,center3):
	'''
	输入一组二维点
	把这些点分为分别以center1,2,3为中心的3类
	并计算分类后的新的中心点
	返回新的3个中心点
	和三个类点的列表
	'''
	#每次更新前要初始化列表
	c1=np.array([0,0])
	c2=np.array([0,0])
	c3=np.array([0,0])
	#print data
	for point in data:
		#print point
		classNum = getClassNum(point,center1,center2,center3)
		if classNum == 1:
			c1=np.vstack((c1,point))
		elif classNum == 2:
			c2=np.vstack((c2,point))
		else:
			c3=np.vstack((c3,point))
	#计算每个类别的中心点
	#print c1[1:-1]
	new1 = getCenter(c1[1:-1])
	new2 = getCenter(c2[1:-1])
	new3 = getCenter(c3[1:-1])

	return new1,new2,new3,c1,c2,c3

def updateDist(new1,new2,new3,old1,old2,old3):
	'''
	计算更新后的3个中心点和原中心点的距离
	返回3个新旧中心点的欧式距离
	'''
	x1 = abs(new1[0] - old1[0])
	y1 = abs(new1[1] - old1[1])
	x2 = abs(new2[0] - old2[0])
	y2 = abs(new2[1] - old2[1])
	x3 = abs(new3[0] - old3[0])
	y3 = abs(new3[1] - old3[1])

	dist1=np.sqrt(x1**2+y1**2)
	dist2=np.sqrt(x2**2+y2**2)
	dist3=np.sqrt(x3**2+y3**2)

	return dist1,dist2,dist3

def k_means(data):
	'''
	data id a np array and have 2 dimonsion like (x,y)
	返回分好类的三个列表
	和3个中心点
	'''
	#随机选3个点
	cn1 = np.random.randint(0,len(data))
	cn2 = np.random.randint(0,len(data))
	while cn2 == cn1:#保证三个点不相等
		cn2 = np.random.randint(0,len(data))
	cn3 = np.random.randint(0,len(data))
	while cn3 == cn2 or cn3 == cn1:
		cn3 = np.random.randint(0,len(data))
	center1 = data[cn1]
	center2 = data[cn2]
	center3 = data[cn3]
	#print center1
	###迭代
	#初始化
	c1=[]
	c2=[]
	c3=[]
	dist1,dist2,dist3 = 10,10,10 #随便设一个比较大的值
	new1,new2,new3 = center1,center2,center3
	#print new1
	iterNum = 0
	while dist1 >= 0.01 or dist2 >= 0.01 or dist3 >= 0.01:
		#保存旧的中心点
		center1,center2,center3 = new1,new2,new3
		#归类并返回新中心点
		new1,new2,new3,c1,c2,c3 = classify(data,new1,new2,new3)
		#print c1
		#计算中心点更新距离
		dist1,dist2,dist3 = updateDist(new1,new2,new3,center1,center2,center3)
		iterNum += 1
		print 'Iteration ',iterNum
		plotPrecess(c1,c2,c3,center1,center2,center3)
	#print type(new1)
	return	c1,c2,c3,new1,new2,new3


def plotPrecess(c1,c2,c3,center1,center2,center3):
	'''
	传入三个类的二维点
	画出图
	'''
	#画布的大小为长8cm高6cm
	plt.figure(figsize=(8,6))
	#s表示点的大小，c就是color，marker就是点的形状,alpha点的亮度，label标签
	plt.scatter(c1[1:,0],c1[1:,1],s=30,c='red',marker='.',alpha=0.5,label='class 1')
	plt.scatter(c2[1:,0],c2[1:,1],s=30,c='green',marker='x',alpha=0.5,label='class 2')
	plt.scatter(c3[1:,0],c3[1:,1],s=30,c='blue',marker='+',alpha=0.5,label='class 3')

	center = []
	center.append(center1)
	center.append(center2)
	center.append(center3)
	center=np.array(center)
	#print center
	plt.scatter(center[:,0],center[:,1],s=30,c='black',marker='o',alpha=1,label='center point')

	plt.title('k_means')
	plt.xlabel('variables x')
	plt.ylabel('variables y')
	plt.legend(loc='upper right')#点的说明
	#plt.show()

def main():
	data = generateData(1000)
	#print data
	c1,c2,c3,center1,center2,center3 = k_means(data)
	#print 'c1:\n',c1,'\n----------------------\nc2:\n',c2,'\n----------------------\nc2:\n',c3
	plotPrecess(c1,c2,c3,center1,center2,center3)
	plt.show()

if __name__ == '__main__':
	main()