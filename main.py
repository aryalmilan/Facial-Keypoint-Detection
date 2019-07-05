from model import my_model
from data_load import load_csv, create_generator
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random

def predict(neural_net,test):
	#randomly selecting images from test folder
	total_images = len(test)
	image_idx = random.sample(range(total_images), 8)
	init = 1
	for i in image_idx :
		image_name = test.iloc[i,0]
		image_path='data/test/' + image_name 
		image_test = load_img(image_path,
							color_mode='grayscale',
							target_size=(224,224)
							)
		img_array = img_to_array(image_test) / 255.0
		img_array= img_array.reshape((1,) + img_array.shape)
		if init:
			image_array= img_array
			init = 0
		else:
			image_array=np.concatenate((image_array,img_array))
	predicted_keypoints=neural_net.predict(image_array)
	return image_idx, predicted_keypoints

def plot_keypoints(csv_file, idx, predicted_keypoints=None):
	ncols=4
	nrows= 4
	fig=plt.gcf()
	fig.set_size_inches(ncols *4 , nrows * 4)
	for i,j  in enumerate(idx):
		image_name=csv_file.iloc[j,0]
		image=mpimg.imread('data/test/'+image_name)
		gt_kpoints=csv_file.iloc[j,1:].as_matrix()
		gt_kpoints=gt_kpoints.astype('float').reshape(-1,2)
		pt_kpoints=predicted_keypoints[i]
		pt_kpoints=pt_kpoints*50+100
		pt_kpoints=pt_kpoints.astype('float').reshape(-1,2)
		sp=plt.subplot(nrows,ncols,i+1)
		sp.axis('Off')
		plt.imshow(image)
		plt.scatter(gt_kpoints[:, 0],gt_kpoints[:, 1], s=20, marker='.', c='m')
		plt.scatter(pt_kpoints[:, 0], pt_kpoints[:, 1], s=40, marker='.', c='g')


#loading our datsets
train_csv='data/training_frames_keypoints.csv'
train_folder='data/training/'
training_dataset = load_csv(train_csv,True)
train_generator = create_generator (train_folder, training_dataset, 64)

#Training Model
neural_net = my_model()
history = neural_net.fit_generator(train_generator,
									epochs = 25,
									verbose = 2)

#Running our model on some testing image dataset
test_csv = 'data/test_frames_keypoints.csv'
test_folder = 'data/test/'
test_dataset=load_csv(test_csv)

#Using our model to predict the keypoints
image_idx, predicted_keypoints = predict(neural_net,test_dataset)
plot_keypoints(test_dataset,image_idx,predicted_keypoints)



