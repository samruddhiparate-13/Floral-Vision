import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model

daisy_dir=os.path.join(r'C:\Users\R S PARATE\Desktop\Python & ML\ML\flowers\daisy')

dandelion_dir=os.path.join(r'C:\Users\R S PARATE\Desktop\Python & ML\ML\flowers\dandelion')

rose_dir=os.path.join(r'C:\Users\R S PARATE\Desktop\Python & ML\ML\flowers\rose')

sunflower_dir=os.path.join(r'C:\Users\R S PARATE\Desktop\Python & ML\ML\flowers\sunflower')

tulip_dir=os.path.join(r'C:\Users\R S PARATE\Desktop\Python & ML\ML\flowers\tulip')


train_tulip_names=os.listdir(tulip_dir)
print(train_tulip_names[:5])

train_sf_names=os.listdir(sunflower_dir)
print(train_sf_names[:5])

img_size = (300, 300)

batch_size=16
train_datagen=ImageDataGenerator(rescale=1./255)
train_generator=train_datagen.flow_from_directory(r'C:\Users\R S PARATE\Desktop\Python & ML\ML\flowers',
                                                  target_size=img_size,#all images resized to 48x48
                                                  batch_size=batch_size,
                                                  color_mode='grayscale',#gray scale means black and white image,rgb means colour image
                                                  classes=['daisy','dandelion','rose','sunflower','tulip'],
                                                  class_mode='categorical')##two classes then binary,more than two classes then categorical
# target_size=(200,200)
##we will try different numbers
model=tf.keras.models.Sequential([
    #Note the input shape is the desired size of the image 
    
    #The first convolution
    tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(img_size[0], img_size[1],1)),
    tf.keras.layers.MaxPooling2D(2,2),
    
    #The second convolution
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    #The third convolution
    tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    #Flatten the results
    tf.keras.layers.Flatten(),
    
    tf.keras.layers.Dense(64,activation='relu'),
    
    #5 output neurons for 5 classes with the softmax activation
    tf.keras.layers.Dense(5,activation='softmax')#predict the probability i.e, it will predict the target image
    ])
model.summary()
model.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),metrics=['acc'])
total_sample=train_generator.n
num_epochs=5
model.fit(train_generator,steps_per_epoch=int(total_sample/batch_size),
                    epochs=num_epochs,verbose=1)
model_json=model.to_json()
with open("modelGG.json","w") as json_file:
    json_file.write(model_json)
model.save_weights("model66.h5")
print("saved model to disk")