import pathlib
import json
import numpy as np
import collections

MODEL_INPUT_IMG_SIZE = (224, 224)

DATA_DIR = pathlib.Path('data')

ASSETS_DIR = DATA_DIR / 'assets'

DS_DIR = DATA_DIR / 'dataset'

OUTPUT_DIR = DATA_DIR / 'output'

VIDS_DIR = DATA_DIR / 'videos'

TRAIN_IDX_2_INIT_DWNLD_PATH = ASSETS_DIR / 'train_idx_2_init_dwnld_idx_w_same_url.json'
TRAIN_INSTANCE_2_CROPPED_PATH = ASSETS_DIR / 'train_instance_2_cropped.json'
TRAIN_INSTANCE_2_TRIMMED_PATH = ASSETS_DIR / 'train_instance_2_trimmed.json'

VAL_IDX_2_INIT_DWNLD_PATH = ASSETS_DIR / 'val_idx_2_init_dwnld_idx_w_same_url.json'
VAL_INSTANCE_2_CROPPED_PATH = ASSETS_DIR / 'val_instance_2_cropped.json'
VAL_INSTANCE_2_TRIMMED_PATH = ASSETS_DIR / 'val_instance_2_trimmed.json'

TEST_IDX_2_INIT_DWNLD_PATH = ASSETS_DIR / 'test_idx_2_init_dwnld_idx_w_same_url.json'
TEST_INSTANCE_2_CROPPED_PATH = ASSETS_DIR / 'test_instance_2_cropped.json'
TEST_INSTANCE_2_TRIMMED_PATH = ASSETS_DIR / 'test_instance_2_trimmed.json'

DS_CLASSES_PATH = DS_DIR / 'MSASL_classes.json'
TRAIN_DATA_PATH = DS_DIR / 'MSASL_train.json'
VAL_DATA_PATH = DS_DIR / 'MSASL_val.json'
TEST_DATA_PATH = DS_DIR / 'MSASL_test.json'

TRAIN_VIDS_DIR = VIDS_DIR / 'train'
TRAIN_VIDS_DWNLD_DIR = TRAIN_VIDS_DIR / 'downloaded'
TRAIN_VIDS_TRIMMED_DIR = TRAIN_VIDS_DIR / 'trimmed'
TRAIN_VIDS_CROPPED_DIR = TRAIN_VIDS_DIR / 'cropped'

VAL_VIDS_DIR = VIDS_DIR / 'val'
VAL_VIDS_DWNLD_DIR = VAL_VIDS_DIR / 'downloaded'
VAL_VIDS_TRIMMED_DIR = VAL_VIDS_DIR / 'trimmed'
VAL_VIDS_CROPPED_DIR = VAL_VIDS_DIR / 'cropped'

TEST_VIDS_DIR = VIDS_DIR / 'test'
TEST_VIDS_DWNLD_DIR = TEST_VIDS_DIR / 'downloaded'
TEST_VIDS_TRIMMED_DIR = TEST_VIDS_DIR / 'trimmed'
TEST_VIDS_CROPPED_DIR = TEST_VIDS_DIR / 'cropped'


with open(DS_CLASSES_PATH, 'r') as f:
    CLASSES = json.load(f)
    f.close()

with open(TRAIN_DATA_PATH, 'r') as f:
    TRAIN_DATA = json.load(f)
    f.close()

with open(VAL_DATA_PATH, 'r') as f:
    VAL_DATA = json.load(f)
    f.close()

with open(TEST_DATA_PATH, 'r') as f:
    TEST_DATA = json.load(f)
    f.close()


FIRST_5_CLASSES = CLASSES[:5]
INSTANCES_PER_CLASS = 10
label_2_nbInstances = collections.defaultdict(list)
train_indices_of_first_5_classes = []
for i, instance in enumerate(TRAIN_DATA):
    if instance['label'] < 5:
        label_ = instance['label']
        tmp = label_2_nbInstances[label_]
        tmp.append(i)
        train_indices_of_first_5_classes.append(i)

train_indices_of_first_5_classes = np.array(train_indices_of_first_5_classes)

val_label_2_nbInstances = collections.defaultdict(list)
val_indices_of_first_5_classes = []
for i, instance in enumerate(VAL_DATA):
    if instance['label'] < 5:
        label_ = instance['label']
        tmp = val_label_2_nbInstances[label_]
        tmp.append(i)
        val_indices_of_first_5_classes.append(i)

val_indices_of_first_5_classes = np.array(val_indices_of_first_5_classes)

test_label_2_nbInstances = collections.defaultdict(list)
test_indices_of_first_5_classes = []
for i, instance in enumerate(TEST_DATA):
    if instance['label'] < 5:
        label_ = instance['label']
        tmp = test_label_2_nbInstances[label_]
        tmp.append(i)
        test_indices_of_first_5_classes.append(i)

test_indices_of_first_5_classes = np.array(test_indices_of_first_5_classes)