import numpy as np
import os
import time

from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

# split data into training and testing set manually
'''
import os, random, shutil
try:
    os.mkdir('dataset/train')
    os.mkdir('dataset/test')
except:
    print("train test directory already exits.")


image_paths = os.listdir('dataset/images')
random.shuffle(image_paths)


for i, image_path in enumerate(image_paths):
  if i < int(len(image_paths) * 0.8):
    try:
        shutil.copy(f'dataset/images/{image_path}', 'dataset/train')
        shutil.copy(f'dataset/annotations/{image_path.replace("jpg", "xml")}', 'dataset/train')
    except:
        shutil.copy(f'dataset/images/{image_path}', 'dataset/train')
        shutil.copy(f'dataset/annotations/{image_path.replace("jpeg", "xml")}', 'dataset/train')
  else:
    try:
        shutil.copy(f'dataset/images/{image_path}', 'dataset/test')
        shutil.copy(f'dataset/annotations/{image_path.replace("jpg", "xml")}', 'dataset/test')
    except:
        shutil.copy(f'dataset/images/{image_path}', 'dataset/test')
        shutil.copy(f'dataset/annotations/{image_path.replace("jpeg", "xml")}', 'dataset/test')
'''

# select model architecture
#spec = model_spec.get('efficientdet_lite0')
spec = object_detector.EfficientDetSpec(
  model_name='efficientdet-lite0',
  uri='https://tfhub.dev/tensorflow/efficientdet/lite0/feature-vector/1', 
  hparams={'max_instances_per_image': 200})

# Load the dataset
train_data = object_detector.DataLoader.from_pascal_voc(images_dir='dataset/train', annotations_dir='dataset/train', label_map=['1head', '1tail', '5head', '5tail', '10head', '10tail'])
validation_data = object_detector.DataLoader.from_pascal_voc(images_dir='dataset/test', annotations_dir='dataset/test', label_map=['1head', '1tail', '5head', '5tail', '10head', '10tail'])

# Training of tensorflow lite model 
model = object_detector.create(train_data, model_spec=spec, epochs = 1000, batch_size=16, train_whole_model=True, validation_data=validation_data, do_train=True)

time.sleep(10)
# Model evaluation on test data
model.evaluate(validation_data)

# Export tensorflow model
model.export(export_dir='./models/coin_best_16_mm')
