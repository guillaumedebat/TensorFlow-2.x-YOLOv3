import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import tensorflow as tf
from yolov3.utils import detect_image, detect_realtime, detect_video, Load_Yolo_model, detect_video_realtime_mp, image_preprocess, image_preprocessUint8
from yolov3.configs import *
import glob



allFiles = []
nb = 200
allCat = glob.glob("./OIDv4_ToolKit/OID/Dataset/train/Cat/*.jpg")
for i in range(nb):
    allFiles.append(allCat[i])

allH = glob.glob("./OIDv4_ToolKit/OID/Dataset/train/Person/*.jpg")
for i in range(nb):
    allFiles.append(allH[i])

allCat = glob.glob("./OIDv4_ToolKit/OID/Dataset/test/Cat/*.jpg")
for i in range(nb):
    allFiles.append(allCat[i])

allH = glob.glob("./OIDv4_ToolKit/OID/Dataset/test/Person/*.jpg")
for i in range(nb):
    allFiles.append(allH[i])



def rep_data_gen2():
    a = []
    for i in range(3*nb):
        inst = allFiles[i]
        image_path = inst
        original_image      = cv2.imread(image_path)
        original_image      = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_image      = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        image_data = image_preprocessUint8(np.copy(original_image), [YOLO_INPUT_SIZE, YOLO_INPUT_SIZE])
        image_data = image_data[np.newaxis, ...].astype(np.uint8) #float32)
        
        a.append(image_data)
    a = np.array(a)
    print(a.shape) # a is np array of 160 3D images
    img = tf.data.Dataset.from_tensor_slices(a).batch(1)
    for i in img.take(128):
        print(i)
        yield [i]

def rep_data_gen():
    a = []
    for i in range(3*nb):
        inst = allFiles[i]
        image_path = inst
        original_image      = cv2.imread(image_path)
        #original_image      = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        #original_image      = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        image_data = image_preprocess(np.copy(original_image), [YOLO_INPUT_SIZE, YOLO_INPUT_SIZE])
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        yield [image_data]

# Reference: https://github.com/sayakpaul/Adventures-in-TensorFlow-Lite/blob/master/DeepLabV3/DeepLab_TFLite_COCO.ipynb
# Load the TensorFlow model
# The preprocessing and the post-processing steps should not be included in the TF Lite model graph 
# because some operations (ArgMax) might not support the delegates. 
# Insepct the graph using Netron https://lutzroeder.github.io/netron/
os.system('clear')
yolo = Load_Yolo_model()
print('end')


# Optional: Perform the simplest optimization known as post-training dynamic range quantization.
# https://www.tensorflow.org/lite/performance/post_training_quantization#dynamic_range_quantization
# You can refer to the same document for other types of optimizations.

converter = tf.lite.TFLiteConverter.from_keras_model(yolo)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
#    tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
    #tf.lite.OpsSet.SELECT_TF_OPS,
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

converter.inference_input_type = tf.int8  # or tf.uint8
converter.inference_output_type = tf.int8  # or tf.uint8
converter.representative_dataset=rep_data_gen
#converter.experimental_enable_resource_variables = True
# Convert to TFLite Model
tflite_model = converter.convert()

with open('test_TinyV4.tflite', 'wb') as f:
  f.write(tflite_model)
