import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.preprocessing import image

model = keras.models.load_model('best_model_v2.keras')

test_image_path = '/mnt/c/Users/Aaditya/Desktop/Coding/Python/ML/CATS VS DOGS/testimagedog.jpeg'

print(test_image_path, "\nI have given the test image of a Dog")

img = image.load_img(test_image_path, target_size=(128, 128))
img_array = image.img_to_array(img)
img_batch = np.expand_dims(img_array, axis=0)
final_image = img_batch / 255.0

prediction = model.predict(final_image)

print("Raw prediction from model:")
print(prediction)

if prediction[0][0] < 0.5:
    print("I think it's a cat!")
else:
    print("I think it's a dog!")

