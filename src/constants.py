import pathlib
import numpy as np
import collections
from utilities import create_dummy_JSON_file, read_dict_from_JSON

MODEL_INPUT_IMG_SIZE = (224, 224)

DATA_DIR = pathlib.Path('data')

ASSETS_DIR = DATA_DIR / 'assets'

if not ASSETS_DIR.exists():
    ASSETS_DIR.mkdir(parents=True)

DS_DIR = DATA_DIR / 'dataset'

OUTPUT_DIR = DATA_DIR / 'output'

VIDS_DIR = DATA_DIR / 'videos'

TRAIN_IDX_2_INIT_DWNLD_PATH = ASSETS_DIR / \
    'train_idx_2_init_dwnld_idx_w_same_url.json'
TRAIN_INSTANCE_2_CROPPED_PATH = ASSETS_DIR / 'train_instance_2_cropped.json'
TRAIN_INSTANCE_2_TRIMMED_PATH = ASSETS_DIR / 'train_instance_2_trimmed.json'

if not TRAIN_IDX_2_INIT_DWNLD_PATH.exists():
    create_dummy_JSON_file(TRAIN_IDX_2_INIT_DWNLD_PATH)
if not TRAIN_INSTANCE_2_CROPPED_PATH.exists():
    create_dummy_JSON_file(TRAIN_INSTANCE_2_CROPPED_PATH)
if not TRAIN_INSTANCE_2_TRIMMED_PATH.exists():
    create_dummy_JSON_file(TRAIN_INSTANCE_2_TRIMMED_PATH)

VAL_IDX_2_INIT_DWNLD_PATH = ASSETS_DIR / \
    'val_idx_2_init_dwnld_idx_w_same_url.json'
VAL_INSTANCE_2_CROPPED_PATH = ASSETS_DIR / 'val_instance_2_cropped.json'
VAL_INSTANCE_2_TRIMMED_PATH = ASSETS_DIR / 'val_instance_2_trimmed.json'

if not VAL_IDX_2_INIT_DWNLD_PATH.exists():
    create_dummy_JSON_file(VAL_IDX_2_INIT_DWNLD_PATH)
if not VAL_INSTANCE_2_CROPPED_PATH.exists():
    create_dummy_JSON_file(VAL_INSTANCE_2_CROPPED_PATH)
if not VAL_INSTANCE_2_TRIMMED_PATH.exists():
    create_dummy_JSON_file(VAL_INSTANCE_2_TRIMMED_PATH)

TEST_IDX_2_INIT_DWNLD_PATH = ASSETS_DIR / \
    'test_idx_2_init_dwnld_idx_w_same_url.json'
TEST_INSTANCE_2_CROPPED_PATH = ASSETS_DIR / 'test_instance_2_cropped.json'
TEST_INSTANCE_2_TRIMMED_PATH = ASSETS_DIR / 'test_instance_2_trimmed.json'

if not TEST_IDX_2_INIT_DWNLD_PATH.exists():
    create_dummy_JSON_file(TEST_IDX_2_INIT_DWNLD_PATH)
if not TEST_INSTANCE_2_CROPPED_PATH.exists():
    create_dummy_JSON_file(TEST_INSTANCE_2_CROPPED_PATH)
if not TEST_INSTANCE_2_TRIMMED_PATH.exists():
    create_dummy_JSON_file(TEST_INSTANCE_2_TRIMMED_PATH)


TRAIN_VIDS_DIR = VIDS_DIR / 'train'
TRAIN_VIDS_DWNLD_DIR = TRAIN_VIDS_DIR / 'downloaded'
TRAIN_VIDS_TRIMMED_DIR = TRAIN_VIDS_DIR / 'trimmed'
TRAIN_VIDS_CROPPED_DIR = TRAIN_VIDS_DIR / 'cropped'
if not TRAIN_VIDS_DWNLD_DIR.exists():
    TRAIN_VIDS_DWNLD_DIR.mkdir(parents=True)
if not TRAIN_VIDS_TRIMMED_DIR.exists():
    TRAIN_VIDS_TRIMMED_DIR.mkdir(parents=True)
if not TRAIN_VIDS_CROPPED_DIR.exists():
    TRAIN_VIDS_CROPPED_DIR.mkdir(parents=True)

VAL_VIDS_DIR = VIDS_DIR / 'val'
VAL_VIDS_DWNLD_DIR = VAL_VIDS_DIR / 'downloaded'
VAL_VIDS_TRIMMED_DIR = VAL_VIDS_DIR / 'trimmed'
VAL_VIDS_CROPPED_DIR = VAL_VIDS_DIR / 'cropped'
if not VAL_VIDS_DWNLD_DIR.exists():
    VAL_VIDS_DWNLD_DIR.mkdir(parents=True)
if not VAL_VIDS_TRIMMED_DIR.exists():
    VAL_VIDS_TRIMMED_DIR.mkdir(parents=True)
if not VAL_VIDS_CROPPED_DIR.exists():
    VAL_VIDS_CROPPED_DIR.mkdir(parents=True)

TEST_VIDS_DIR = VIDS_DIR / 'test'
TEST_VIDS_DWNLD_DIR = TEST_VIDS_DIR / 'downloaded'
TEST_VIDS_TRIMMED_DIR = TEST_VIDS_DIR / 'trimmed'
TEST_VIDS_CROPPED_DIR = TEST_VIDS_DIR / 'cropped'
if not TEST_VIDS_DWNLD_DIR.exists():
    TEST_VIDS_DWNLD_DIR.mkdir(parents=True)
if not TEST_VIDS_TRIMMED_DIR.exists():
    TEST_VIDS_TRIMMED_DIR.mkdir(parents=True)
if not TEST_VIDS_CROPPED_DIR.exists():
    TEST_VIDS_CROPPED_DIR.mkdir(parents=True)


# Will be created anyways if not existant when fitting the model
OPTFLOW_LOGS_DIR = pathlib.Path('optFlow_logs')
OPTFLOW_MODEL_DIR = pathlib.Path('optFlow_model')
RGB_LOGS_DIR = pathlib.Path('rgb_logs')
RGB_MODEL_DIR = pathlib.Path('rgb_model')
# /Will be created anyways if not existant when fitting the model

# Must exist
DS_CLASSES_PATH = DS_DIR / 'MSASL_classes.json'
TRAIN_DATA_PATH = DS_DIR / 'MSASL_train.json'
VAL_DATA_PATH = DS_DIR / 'MSASL_val.json'
TEST_DATA_PATH = DS_DIR / 'MSASL_test.json'
# /Must exist

CLASSES = read_dict_from_JSON(DS_CLASSES_PATH)
TRAIN_DATA = read_dict_from_JSON(TRAIN_DATA_PATH)
VAL_DATA = read_dict_from_JSON(VAL_DATA_PATH)
TEST_DATA = read_dict_from_JSON(TEST_DATA_PATH)

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