import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Loading csv file that consisits of filenam and facial keypoints in each file
def load_csv(csv_file,normalize=False):
	dataset = pd.read_csv(csv_file)
	#keypoints to be centered around 0 with range of [-1,1]
	if normalize:
		for  i in range(len(dataset)):
			dataset.iloc[i,1:]=(dataset.iloc[i,1:]-100)/50
	return dataset

def create_generator(img_folder, data_frame,batch_size):
	data_generator=ImageDataGenerator (rescale = 1./255)
	generator=data_generator.flow_from_dataframe(dataframe = data_frame,
												directory = img_folder,
												x_col = 'Unnamed: 0',
												y_col =[str(i) for i in range(136)],
												color_mode ='grayscale',
												class_mode ='other',
												target_size = (224,224),
												batch_size = batch_size)
	return generator
