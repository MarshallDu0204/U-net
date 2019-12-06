import os,shutil
from PIL import Image
import Augmentor
import numpy as np
import cv2
import random
import tensorflow as tf

train_set_writer = tf.python_io.TFRecordWriter("/root/train_set_Unet.tfrecords")

def resize():
	path = "C:/Users/24400/Desktop/newMask"
	imageList = os.listdir(path)
	i = 0
	for image in imageList:
		os.mkdir("C:/Users/24400/Desktop/mask/"+str(i))
		shutil.copy(path+"/"+image,"C:/Users/24400/Desktop/mask/"+str(i))
		i+=1

def augment(inputPath,outputPath):
    os.mkdir(outputPath)
    p = Augmentor.Pipeline(
        source_directory=inputPath,
        output_directory=outputPath
    )
    p.rotate(probability=0.5, max_left_rotation=2, max_right_rotation=2)
    p.zoom(probability=0.5, min_factor=1.1, max_factor=1.2)
    p.skew(probability=0.5)
    p.random_distortion(probability=0.5, grid_width=96, grid_height=96, magnitude=1)
    p.shear(probability=0.2, max_shear_left=2, max_shear_right=2)
    #p.crop_random(probability=0.2, percentage_area=0.8)
    p.flip_random(probability=0.2)
    p.sample(n=40)


def excAug():
	path = "C:/Users/24400/Desktop/mask1"
	imageList = os.listdir(path)
	i = 0
	while i!=len(imageList):
		augment(path+"/"+str(i),"C:/Users/24400/Desktop/mask"+"/"+str(i))
		i+=1

def processImg(image_row):
	image_data = []
	for x in image_row:
		tempData = []
		for y in x:
			tempData.append(y)
		image_data.append(tempData)
	image_data = np.asarray(image_data,dtype = np.uint8)
	image_data[image_data<170] = 0
	image_data[image_data>=170] = 1
	return image_data


def constructList():
	path = "/root/data"
	listNum = len(os.listdir(path))
	dataNum = len(os.listdir(path+"/"+str(0)))

	totalNum = listNum*dataNum
	dataList = []
	i=0
	while i!=totalNum:
		dataList.append(i)
		i+=1
	shuffleData(dataList)
	dataPathList,maskPathList = readImage(dataList)
	i = 0
	while i!=len(dataPathList):
		feature = {}
		data_row = Image.open(dataPathList[i])
		data_row = np.asarray(data_row)

		feature['img'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[data_row.tobytes()]))

		mask_row = Image.open(maskPathList[i])
		mask_row = np.asarray(mask_row)

		mask_row = processImg(mask_row)
		feature['label'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[mask_row.tobytes()]))
		example = tf.train.Example(features=tf.train.Features(feature=feature))
		train_set_writer.write(example.SerializeToString())
		print(i,totalNum)
		i+=1
	


def shuffleData(dataList):
	random.shuffle(dataList)
	random.shuffle(dataList)
	random.shuffle(dataList)
	return dataList

def readImage(dataList):
	dataPathList = []
	maskPathList = []
	for index in dataList:
		folderNum = index//40
		imgNum = index-folderNum*40
		dataPath = "/root/data/"+str(folderNum)
		imgList = os.listdir(dataPath)
		imgPath = dataPath+"/"+str(imgList[imgNum])
		dataPathList.append(imgPath)
		maskPath = "/root/mask/"+str(folderNum)
		maskList = os.listdir(maskPath)
		maskPath = maskPath+"/"+str(maskList[imgNum])
		maskPathList.append(maskPath)
	return dataPathList,maskPathList


constructList()


