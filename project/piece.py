import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2

#read data
data_dir = '/home/su/code/bio_Hw/chest_xray/chest_xray/'

train_dir = data_dir+'train/'
test_dir = data_dir+'test/'

train_normal_dir = train_dir+'NORMAL/'
train_pneumonia_dir = train_dir+'PNEUMONIA/'

normal_cases = glob.glob(train_normal_dir+'*.jpeg')
#print(normal_cases)
pneumonia_cases = glob.glob(train_pneumonia_dir+'*.jpeg')

train_data = []
for img in normal_cases:
	train_data.append((img,0))
# print(train_data[0])
for img in pneumonia_cases:
	train_data.append((img,1))

img = cv2.imread(train_data[0][0])
gray_img = cv2.cvtColor(train_data[0][0],cv2.COLOR_BGR2GRAY)
cv2.imshow('1',gray_img)
cv2.waitKey(0)

# def main():
# 	a=1 if 1==1 else 0
# 	print a
# if __name__ == '__main__':
# 	main()

