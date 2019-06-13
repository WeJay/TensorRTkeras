# force reset ipython namespaces
# %reset -f
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
from pathlib import Path
import time
import datetime
import pathlib
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import load_model
import os
import shutil
import json
import cv2
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 1.0
#config.log_device_placement = True
sess = tf.Session(config=config)
set_session(sess) 

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
modelName = ''                        #model_name
model = load_model('Fine-tuning-'+modelName+'.h5')

total_time = 0; n_time_inference = 50
image = cv2.imread("4751/R1_10_1560219311.859064.bmp")
start_predict_image = time.time()
npdata = []
imgOrg = cv2.resize(image, (192,96), interpolation=cv2.INTER_NEAREST)
img = cv2.cvtColor(imgOrg, cv2.COLOR_BGR2RGB)
npdata.append(img / 255)
inputimage = np.asarray(npdata)
predictions = model.predict(inputimage)
for i in range(n_time_inference):
  t1 = time.time()
  predictions = model.predict(inputimage)
  t2 = time.time()
  delta_time = t2 - t1
  total_time += delta_time
  print("needed time in inference-" + str(i) + ": ", delta_time)
avg_time_tensorRT = total_time / n_time_inference
print('average(sec):{},fps:{}'.format(avg_time_tensorRT,1/avg_time_tensorRT))

