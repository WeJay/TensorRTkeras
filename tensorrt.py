# force reset ipython namespaces
# %reset -f
# .h5 to .pb

import tensorflow as tf
from tensorflow.python.framework import graph_io
from tensorflow.keras.models import load_model


# Clear any previous session.
tf.keras.backend.clear_session()

save_pb_dir = '/model'
model_fname = '/*.h5'
def freeze_graph(graph, session, output, save_pb_dir='.', save_pb_name='frozen_model.pb', save_pb_as_text=False):
    with graph.as_default():
        graphdef_inf = tf.graph_util.remove_training_nodes(graph.as_graph_def())
        graphdef_frozen = tf.graph_util.convert_variables_to_constants(session, graphdef_inf, output)
        graph_io.write_graph(graphdef_frozen, save_pb_dir, save_pb_name, as_text=save_pb_as_text)
        return graphdef_frozen

# This line must be executed before loading Keras model.
tf.keras.backend.set_learning_phase(0) 

model = load_model(model_fname)

session = tf.keras.backend.get_session()

input_names = [t.op.name for t in model.inputs]
output_names = [t.op.name for t in model.outputs]

# Prints input and output nodes names, take notes of them.
print(input_names, output_names)

frozen_graph = freeze_graph(session.graph, session, [out.op.name for out in model.outputs], save_pb_dir=save_pb_dir)

# Optimize with TensorRT

# Import the needed libraries
# %reset -f
import cv2
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import tensorrt as trt
#import tensorflow.contrib.tensorrt as trt
from tensorflow.python.platform import gfile
# function to read a ".pb" model 
# (can be used to read frozen model or TensorRT model)
def read_pb_graph(model):
  with gfile.FastGFile(model,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
  return graph_def
frozen_graph = read_pb_graph("./model/frozen_model_AE.pb")

your_outputs = ["dense_1/Softmax:0"]
# convert (optimize) frozen model to TensorRT model
trt_graph = trt.create_inference_graph(
    is_dynamic_op=True,
    input_graph_def=frozen_graph,# frozen model
    outputs=your_outputs,
    max_batch_size=1,# specify your max batch size
    max_workspace_size_bytes=4*(10**9),# specify the max workspace
    precision_mode="FP32") # precision, can be "FP32" (32 floating point precision) or "FP16"

#write the TensorRT model to be used later for inference
with gfile.FastGFile("./model/trt_5.0.2_FP32.pb", 'wb') as f:
    f.write(trt_graph.SerializeToString())
print("TensorRT model is successfully stored!")
# check how many ops of the original frozen model
all_nodes = len([1 for n in frozen_graph.node])
print("numb. of all_nodes in frozen graph:", all_nodes)

# check how many ops that is converted to TensorRT engine
trt_engine_nodes = len([1 for n in trt_graph.node if str(n.op) == 'TRTEngineOp'])
print("numb. of trt_engine_nodes in TensorRT graph:", trt_engine_nodes)
all_nodes = len([1 for n in trt_graph.node])
print("numb. of all_nodes in TensorRT graph:", all_nodes)

graph_io.write_graph(trt_graph, "./model/",
                     "trt_graph.pb", as_text=False)

