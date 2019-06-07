from __future__ import division
from __future__ import print_function


from DataLoader import DataLoader, Batch
from Model import Model, DecoderType
from SamplePreprocessor import preprocess
from WordSegmentation import wordSegmentation, prepareImg

import shutil
import os
import argparse
import editdistance
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

from ocr import page, words
from ocr.helpers import implt, resize

#%matplotlib inline
plt.rcParams['figure.figsize'] = (15.0, 10.0)

def main():
	Infilename = sys.argv[1]

	image = cv2.cvtColor(cv2.imread(Infilename), cv2.COLOR_BGR2RGB)
	crop = page.detection(image)
	boxes = words.detection(crop)
	lines = words.sort_words(boxes)
	crop = cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY)
	imLines=[]
	for line in lines:
		imLine=[]
		for (x1, y1, x2, y2) in line:
			imLine.append(crop[y1:y2, x1:x2])
		imLines.append(imLine)
	

	decoderType = DecoderType.WordBeamSearch
	#decoderType = DecoderType.BeamSearch
	#decoderType = DecoderType.BestPath
	

	model = Model(open('../model/charList.txt').read(), decoderType, mustRestore=True)
	file1 = open("myfile.txt","w") 
	recognizedL = []
  
	print("-------------------Predicted Handwritten Text-------------------------")
	for line in imLines:
		imgs = []
		for word in line:
			imgs.append(preprocess(word, Model.imgSize))
		batch = Batch(None, imgs)
		(recognized, probability) = model.inferBatch(batch, True)
		l = ""
		for pw in recognized:
			l += pw
			l += ' '
			print(pw, end=" ")
		print()
		l += '\n'
		recognizedL.append(l)
	file1.writelines(recognizedL) 
	file1.close()



if __name__ == '__main__':
	main()



