# force reset ipython namespaces
# %reset -f
# import the needed libraries
import tensorflow as tf
#import tensorflow.contrib.tensorrt as trt # must import this although we will not use it explicitly
from tensorflow.contrib import tensorrt as trt
from tensorflow.python.platform import gfile
from PIL import Image
import numpy as np
import time
from matplotlib import pyplot as plt
import cv2

# function to read a ".pb" model 
# (can be used to read frozen model or TensorRT model)
def read_pb_graph(model):
  with gfile.FastGFile(model,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
  return graph_def

# variable
TENSORRT_MODEL_PATH = 'model/trt_graph.pb'
output_names = ['dense_1/Softmax']
input_names = ['input_1']
input_tensor_name = input_names[0] + ":0"
output_tensor_name = output_names[0] + ":0"


image = cv2.imread("4751/R1_10_1560219311.859064.bmp")
npdata = []
imgOrg = cv2.resize(image, (192,96), interpolation=cv2.INTER_NEAREST)
img = cv2.cvtColor(imgOrg, cv2.COLOR_BGR2RGB)
npdata.append(img / 255)
inputimage = np.asarray(npdata)


graph = tf.Graph()
with graph.as_default():
    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5))) as sess:
        # read TensorRT model
        trt_graph = read_pb_graph(TENSORRT_MODEL_PATH)

        # obtain the corresponding input-output tensor
        tf.import_graph_def(trt_graph, name='')
        input = sess.graph.get_tensor_by_name(input_tensor_name)
        output = sess.graph.get_tensor_by_name(output_tensor_name)

        # in this case, it demonstrates to perform inference for 50 times
        total_time = 0; n_time_inference = 50
        out_pred = sess.run(output, feed_dict={input: inputimage})
        for i in range(n_time_inference):
            t1 = time.time()
            out_pred = sess.run(output, feed_dict={input: inputimage})
            t2 = time.time()
            delta_time = t2 - t1
            total_time += delta_time
            print("needed time in inference-" + str(i) + ": ", delta_time)
        avg_time_tensorRT = total_time / n_time_inference       
        print('average(sec):{},fps:{}'.format(avg_time_tensorRT,1/avg_time_tensorRT))