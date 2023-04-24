import os
import cv2
from PIL import Image 
import numpy as np 
from patchify import patchify
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from matplotlib import pyplot as plt
import random
import wandb
from wandb.keras import WandbCallback

from keras.models import load_model 

minmaxscaler = MinMaxScaler()

wandb.init(project="satellite-segmentation-dubai-images", entity="prodramp")

wandb.log({'accuracy': jaccard_coef, 'loss': total_loss})
     

wandb.config.dropout = 0.2
     

model_history = model.fit(X_train, y_train,
                          batch_size=16,
                          verbose=1,
                          epochs=10,
                          validation_data=(X_test, y_test),
                          callbacks=[WandbCallback()],
                          shuffle=False)
                   
                   
history_a = model_history

loss = history_a.history['loss']
val_loss = history_a.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label="Training Loss")
plt.plot(epochs, val_loss, 'r', label="Validation Loss")
plt.title("Training Vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

jaccard_coef = history_a.history['jaccard_coef']
val_jaccard_coef = history_a.history['val_jaccard_coef']

epochs = range(1, len(jaccard_coef) + 1)
plt.plot(epochs, jaccard_coef, 'y', label="Training IoU")
plt.plot(epochs, val_jaccard_coef, 'r', label="Validation IoU")
plt.title("Training Vs Validation IoU")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Model saving and reloading

model.save('satellite-imagery.h5')
saved_model = load_model('/content/satellite-imagery.h5',
                         custom_objects=({'dice_loss_plus_1focal_loss': total_loss, 
                                          'jaccard_coef': jaccard_coef}))
                                          
                                          
                                          

# Prediction using the custom image from Google Map
plt.imshow(Image.open('/content/fc-img1.jpg'))

plt.imshow(Image.open('/content/fc-img2.jpg'))

image = Image.open('/content/fc-img1.jpg')
image = image.resize((256,256))
image = np.array(image)
image = np.expand_dims(image, 0)

prediction = saved_model.predict(image)

predicted_image = np.argmax(prediction, axis=3)
predicted_image = predicted_image[0,:,:]


plt.figure(figsize=(14,8))
plt.subplot(231)
plt.title("Original Image")
plt.imshow(Image.open('/content/fc-img1.jpg'))
plt.subplot(232)
plt.title("Predicted Image")
plt.imshow(predicted_image)


