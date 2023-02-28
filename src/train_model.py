import tensorflow as tf

from frame_generator import FrameGenerator
from optical_flow_generator import OpticalFlowStream
import constants

# Old implementation where preprocessing was done separately and not on the fly when feeding the vids to the model
# According to https://stackoverflow.com/questions/13780907/is-it-possible-to-np-concatenate-memory-mapped-files
# concatenating memapped arrays without loading them into memory occurs only in pairs

output_signature1 = (
    tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32),
    tf.TensorSpec(shape=(), dtype=tf.int16)
)
# train_ds_1 = tf.data.Dataset.from_generator(
#     FrameGenerator(constants.TRAIN_VIDS_CROPPED_DIR, constants.TRAIN_DATA,
#                    training=True, instance_idx=constants.train_indices_of_first_5_classes),
#     output_signature=output_signature1
# )
output_signature2 = (
    tf.TensorSpec(shape=(None, None, None, 2), dtype=tf.float32),
    tf.TensorSpec(shape=(), dtype=tf.int16)
)
train_ds_2 = tf.data.Dataset.from_generator(
    OpticalFlowStream(constants.TRAIN_VIDS_CROPPED_DIR, constants.TRAIN_DATA,
                      training=True, instance_idx=constants.train_indices_of_first_5_classes),
    output_signature=output_signature2
)

# for frames, labels in train_ds_1.take(5):
#     print(frames.shape, labels)


NUM_VIDS_PER_BATCH = 4
AUTOTUNE = tf.data.AUTOTUNE

# train_ds_1 = train_ds_1.cache().shuffle(300).prefetch(buffer_size=AUTOTUNE)
train_ds_2 = train_ds_2.cache().shuffle(300).prefetch(buffer_size=AUTOTUNE)

# train_ds_1 = train_ds_1.batch(NUM_VIDS_PER_BATCH)
train_ds_2 = train_ds_2.batch(NUM_VIDS_PER_BATCH)

# train_frames, train_labels = next(iter(train_ds_1))
# print(f'Shape of training set of frames: {train_frames.shape}')
# print(f'Shape of training labels: {train_labels.shape}')


# val_ds_1 = tf.data.Dataset.from_generator(
#     FrameGenerator(constants.VAL_VIDS_CROPPED_DIR, constants.VAL_DATA,
#                    training=False, instance_idx=constants.val_indices_of_first_5_classes),
#     output_signature=output_signature1
# )

val_ds_2 = tf.data.Dataset.from_generator(
    OpticalFlowStream(constants.VAL_VIDS_CROPPED_DIR, constants.VAL_DATA,
                      training=False, instance_idx=constants.val_indices_of_first_5_classes),
    output_signature=output_signature2
)

# for frames, labels in val_ds_1.take(5):
#     print(frames.shape, labels)


NUM_VIDS_PER_BATCH = 4
AUTOTUNE = tf.data.AUTOTUNE

# val_ds_1 = val_ds_1.cache().shuffle(300).prefetch(buffer_size=AUTOTUNE)
val_ds_2 = val_ds_2.cache().shuffle(300).prefetch(buffer_size=AUTOTUNE)

# val_ds_1 = val_ds_1.batch(NUM_VIDS_PER_BATCH)
val_ds_2 = val_ds_2.batch(NUM_VIDS_PER_BATCH)

# val_frames, val_labels = next(iter(val_ds_1))
# print(f'Shape of training set of frames: {val_frames.shape}')
# print(f'Shape of training labels: {val_labels.shape}')

from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam

from model import Inception_Inflated3d, freeze_layers, add_top_layer, model_chkpts

rgb_model = Inception_Inflated3d(include_top=False,
                                 weights='rgb_imagenet_and_kinetics',
                                 input_shape=(74, *constants.MODEL_INPUT_IMG_SIZE, 3),
                                 classes=5
                                 )

# print('len(my_model.layers):', len(my_model.layers))
# print('my_model.summary():', my_model.summary())

rgb_model = freeze_layers(rgb_model)
rgb_model = add_top_layer(base_model=rgb_model, classes=5, dropout_prob=0.5)
rgb_model.summary()

rgb_optimizer = Adam(learning_rate=0.0001, decay=1e-6)
rgb_model.compile(optimizer=rgb_optimizer,
                  loss=SparseCategoricalCrossentropy(),  # since labels are ints
                  metrics=['accuracy']
                  )
rgb_model_checkpoints = model_chkpts(log_dir='rgb_logs', model_dir='rgb_model')

# rgb_model.fit(train_ds_1, epochs=10, validation_data=val_ds_1, callbacks=rgb_model_checkpoints)


optFlow_model = Inception_Inflated3d(include_top=False,
                                 weights='flow_imagenet_and_kinetics',
                                 input_shape=(73, *constants.MODEL_INPUT_IMG_SIZE, 2),
                                 classes=5
                                 )

# print('len(my_model.layers):', len(my_model.layers))
# print('my_model.summary():', my_model.summary())

optFlow_model = freeze_layers(optFlow_model)
optFlow_model = add_top_layer(base_model=optFlow_model, classes=5, dropout_prob=0.5)
optFlow_model.summary()

optFlow_optimizer = Adam(learning_rate=0.0001, decay=1e-6)
optFlow_model.compile(optimizer=optFlow_optimizer,
                  loss=SparseCategoricalCrossentropy(),  # since labels are ints
                  metrics=['accuracy']
                  )
optFlow_model_checkpoints = model_chkpts(log_dir='optFlow_logs', model_dir='optFlow_model')

optFlow_model.fit(train_ds_2, epochs=10, validation_data=val_ds_2, callbacks=optFlow_model_checkpoints)
