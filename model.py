import tensorflow as tf

def my_model():
	model=tf.keras.models.Sequential([
		tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(224,224,1)),
    	tf.keras.layers.MaxPooling2D(2,2),
    	tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    	tf.keras.layers.MaxPooling2D(2,2),
    	tf.keras.layers.Conv2D(16,(3,3),activation='relu'),
    	tf.keras.layers.MaxPooling2D(2,2),
    	tf.keras.layers.Flatten(),
    	tf.keras.layers.Dense(1024,activation='relu'),
    	tf.keras.layers.Dense(512,activation='relu'),
    	tf.keras.layers.Dropout(0.5),
    	tf.keras.layers.Dense(136)  
		])
	model.compile(optimizer=tf.train.AdamOptimizer(),
              loss= tf.keras.losses.Huber(),
              metrics=['accuracy'])
	return model
