from __future__ import division
from __future__ import print_function

import sys
import argparse
import cv2
import editdistance
from DataLoader import DataLoader, Batch
from Model import Model, DecoderType
from SamplePreprocessor import preprocess
import os
from WordSegmentation import wordSegmentation, prepareImg
import shutil
import numpy as np


path = '../SegWords'



def infer(model, fnImg):
	"recognize text in image provided by file path"
	img = preprocess(cv2.imread(fnImg, cv2.IMREAD_GRAYSCALE), Model.imgSize)
	batch = Batch(None, [img])
	(recognized, probability) = model.inferBatch(batch, True)
	return (recognized[0], probability)

def main():
	Infilename = sys.argv[1]
	img = cv2.imread(Infilename, cv2.IMREAD_GRAYSCALE)

	# increase contrast
	pxmin = np.min(img)
	pxmax = np.max(img)
	imgContrast = (img - pxmin) / (pxmax - pxmin) * 255

	# increase line width
	kernel = np.ones((3, 3), np.uint8)
	imgMorph = cv2.erode(imgContrast, kernel, iterations = 1)


	img = prepareImg(imgMorph, 50)
	res = wordSegmentation(img, kernelSize=25, sigma=11, theta=7, minArea=100)
	if os.path.isdir(path):
		shutil.rmtree(path)
	os.mkdir(path)
	
	for (j, w) in enumerate(res):
		(wordBox, wordImg) = w
		cv2.imwrite('../SegWords/%d.png'%j, wordImg) 

	files = []
	for filename in sorted(os.listdir(path)):
		files.append(os.path.join(path,filename))
	
	decoderType = DecoderType.WordBeamSearch
	#decoderType = DecoderType.BeamSearch
	#decoderType = DecoderType.BestPath
	

	model = Model(open('../model/charList.txt').read(), decoderType, mustRestore=True)
	imgs = []
	for fp in files:
		imgs.append(preprocess(cv2.imread(fp, cv2.IMREAD_GRAYSCALE), Model.imgSize))
	batch = Batch(None, imgs)
	(recognized, probability) = model.inferBatch(batch, True)

	model = Model(open('../model/charList.txt').read(), decoderType, mustRestore=True)
	file1 = open("myfile.txt","w") 
	l=''
	print('The predicted sentence is : ',end="'")
	for pw in recognized:
		l += pw
		l += ' '
		print(pw, end=" ")
	print("'")
	l += '\n'
	file1.write(l) 
	file1.close()
	# print('The average probability is : ',end="")
	# sum = 0
	# for prob in probability:
	# 	sum += prob
	# print(sum/len(files)*100)


if __name__ == '__main__':
	main()



