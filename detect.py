from keras.applications import InceptionV3
from keras.applications import imagenet_utils
from tensorflow.keras.utils import img_to_array, load_img
from keras.applications.inception_v3 import preprocess_input
import numpy as np
import cv2


img_path = './images/img6.jpg'
img = load_img(img_path)

img = img.resize((299,299))

img_array = img_to_array(img)


img_array = np.expand_dims(img_array, axis=0)


img_array = preprocess_input(img_array)


pretrained_model = InceptionV3(weights="imagenet")


prediction = pretrained_model.predict(img_array)

actual_prediction = imagenet_utils.decode_predictions(prediction)

print("predicted object is:")
print(actual_prediction[0][0][1])
print("with accuracy")
print(actual_prediction[0][0][2]*100)


disp_img = cv2.imread(img_path)

cv2.putText(disp_img, actual_prediction[0][0][1], (15,15),cv2.FONT_HERSHEY_TRIPLEX , 0.8, (0,0,255))


cv2.imshow("Prediction",disp_img)
cv2.waitKey(0)