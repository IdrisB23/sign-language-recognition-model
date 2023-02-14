import os
import pathlib
import json
import collections
import numpy as np
import tensorflow as tf

from download_cropped_vids import dwnld_trim_crop_pipeline_indices, dwnld_trim_crop_pipeline_range
from frame_generator import FrameGenerator
from optical_flow_generator import OpticalFlowStream

from model import my_model_frozen_w_top_layer, model_checkpoints


DATA_DIR = 'data'

DATASET_DIR = os.path.join(DATA_DIR, 'dataset')
DS_CLASSES_PATH = os.path.join(DATASET_DIR, 'MSASL_classes.json')
DS_SYNONYMS_DATA_PATH = os.path.join(DATASET_DIR, 'MSASL_synonyms.json')
TEST_DATA_PATH = os.path.join(DATASET_DIR, 'MSASL_test.json')
TRAIN_DATA_PATH = os.path.join(DATASET_DIR, 'MSASL_train.json')
NEW_TRAIN_DATA_PATH = 'cleansed_train_ds.json'
VAL_DATA_PATH = os.path.join(DATASET_DIR, 'MSASL_val.json')

with open(DS_CLASSES_PATH, 'r') as f:
    CLASSES = json.load(f)
    f.close()

with open(TRAIN_DATA_PATH, 'r') as f:
    TRAINING_DATA = json.load(f)
    f.close()

with open(VAL_DATA_PATH, 'r') as f:
    VAL_DATA = json.load(f)
    f.close()

FIRST_5_CLASSES = CLASSES[:5]
INSTANCES_PER_CLASS = 10
label_2_nbInstances = collections.defaultdict(list)
indices_of_first_5_classes = []
for i, instance in enumerate(TRAINING_DATA):
    if instance['label'] < 5:
        label_ = instance['label']
        tmp = label_2_nbInstances[label_]
        if len(tmp) < INSTANCES_PER_CLASS:
            tmp.append(i)
            indices_of_first_5_classes.append(i)
print(label_2_nbInstances)

print("len(indices_of_first_5_classes):", len(indices_of_first_5_classes))
print(indices_of_first_5_classes)
indices_of_first_5_classes = np.array(indices_of_first_5_classes)
# np.save(INDICES_INSTANCES_OF_SPECIFIC_CLASSES_FILE_PATH, indices_of_first_5_classes)
# vids_w_size_over_threshold = dwnld_trim_crop_pipeline_indices(
#     TRAINING_DATA, '', indices=indices_of_first_5_classes)
# vids_w_size_over_threshold = dwnld_trim_crop_pipeline_range(
#     TRAINING_DATA, 0, len(TRAINING_DATA))
# print(vids_w_size_over_threshold)

# Old implementation where preprocessing was done separately and not on the fly when feeding the vids to the model
# According to https://stackoverflow.com/questions/13780907/is-it-possible-to-np-concatenate-memory-mapped-files
# concatenating memapped arrays without loading them into memory occurs only in pairs

data_pathlib = pathlib.Path('data')
output_pathlib = data_pathlib / 'output'
train_videos_pathlib = data_pathlib / 'videos' / 'train'
val_videos_pathlib = data_pathlib / 'videos' / 'val'
frames_pathlib = output_pathlib / 'frames'
cropped_pathlib = train_videos_pathlib / 'cropped'
val_cropped_pathlib = val_videos_pathlib / 'cropped'
output_signature1 = (
    tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32),
    tf.TensorSpec(shape=(), dtype=tf.int16)
)
train_ds_1 = tf.data.Dataset.from_generator(
    FrameGenerator(cropped_pathlib, TRAINING_DATA,
                   training=True, instance_idx=indices_of_first_5_classes),
    output_signature=output_signature1
)
output_signature2 = (
    tf.TensorSpec(shape=(None, None, None, 2), dtype=tf.float32),
    tf.TensorSpec(shape=(), dtype=tf.int16)
)
train_ds_2 = tf.data.Dataset.from_generator(
    OpticalFlowStream(cropped_pathlib, TRAINING_DATA,
                      training=True, instance_idx=indices_of_first_5_classes),
    output_signature=output_signature2
)

for frames, labels in train_ds_1.take(5):
    print(frames.shape, labels)


NUM_VIDS_PER_BATCH = 4
AUTOTUNE = tf.data.AUTOTUNE

train_ds_1 = train_ds_1.cache().shuffle(300).prefetch(buffer_size=AUTOTUNE)
train_ds_2 = train_ds_2.cache().shuffle(300).prefetch(buffer_size=AUTOTUNE)

train_ds_1 = train_ds_1.batch(NUM_VIDS_PER_BATCH)
train_ds_2 = train_ds_2.batch(NUM_VIDS_PER_BATCH)

train_frames, train_labels = next(iter(train_ds_1))
print(f'Shape of training set of frames: {train_frames.shape}')
print(f'Shape of training labels: {train_labels.shape}')

val_label_2_nbInstances = collections.defaultdict(list)
val_indices_of_first_5_classes = []
for i, instance in enumerate(VAL_DATA):
    if instance['label'] < 5:
        label_ = instance['label']
        tmp = val_label_2_nbInstances[label_]
        tmp.append(i)
        val_indices_of_first_5_classes.append(i)

val_indices_of_first_5_classes = np.array(val_indices_of_first_5_classes)
print(len(val_indices_of_first_5_classes))
# vids_w_size_over_threshold = dwnld_trim_crop_pipeline_indices(
#     val_indices_of_first_5_classes)


val_ds_1 = tf.data.Dataset.from_generator(
    FrameGenerator(val_cropped_pathlib, VAL_DATA,
                   training=False, instance_idx=val_indices_of_first_5_classes),
    output_signature=output_signature1
)

val_ds_2 = tf.data.Dataset.from_generator(
    OpticalFlowStream(val_cropped_pathlib, VAL_DATA,
                   training=False, instance_idx=val_indices_of_first_5_classes),
    output_signature=output_signature2
)

for frames, labels in val_ds_1.take(5):
    print(frames.shape, labels)


NUM_VIDS_PER_BATCH = 4
AUTOTUNE = tf.data.AUTOTUNE

val_ds_1 = val_ds_1.cache().shuffle(300).prefetch(buffer_size=AUTOTUNE)
val_ds_2 = val_ds_2.cache().shuffle(300).prefetch(buffer_size=AUTOTUNE)

val_ds_1 = val_ds_1.batch(NUM_VIDS_PER_BATCH)
val_ds_2 = val_ds_2.batch(NUM_VIDS_PER_BATCH)

val_frames, val_labels = next(iter(val_ds_1))
print(f'Shape of training set of frames: {val_frames.shape}')
print(f'Shape of training labels: {val_labels.shape}')

my_model_frozen_w_top_layer.fit(train_ds_1, epochs=10, validation_data=val_ds_1, callbacks=model_checkpoints)
