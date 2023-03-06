import pathlib

import constants
from utilities import read_dict_from_JSON
from video_augmentation import augment_video
import download_cropped_vids

NUM_AUGMENTATIONS_PER_VID = 10

TRAIN_INSTANCE_2_CROPPED = read_dict_from_JSON(constants.TRAIN_INSTANCE_2_CROPPED_PATH)
VAL_INSTANCE_2_CROPPED = read_dict_from_JSON(constants.VAL_INSTANCE_2_CROPPED_PATH)
TEST_INSTANCE_2_CROPPED = read_dict_from_JSON(constants.TEST_INSTANCE_2_CROPPED_PATH)

# train_vids_w_size_over_threshold = download_cropped_vids.dwnld_trim_crop_pipeline_indices(constants.train_indices_of_first_5_classes, 'train')
# print(train_vids_w_size_over_threshold)
# val_vids_w_size_over_threshold = download_cropped_vids.dwnld_trim_crop_pipeline_indices(constants.val_indices_of_first_5_classes, 'val')
# print(val_vids_w_size_over_threshold)
# test_vids_w_size_over_threshold = download_cropped_vids.dwnld_trim_crop_pipeline_indices(constants.test_indices_of_first_5_classes, 'test')
# print(test_vids_w_size_over_threshold)

for i in constants.train_indices_of_first_5_classes:
    dict_query_result = TRAIN_INSTANCE_2_CROPPED.get(str(i), None)
    if dict_query_result is None:
        continue
    train_vid_pathlib = pathlib.Path(dict_query_result)
    parent_dir = train_vid_pathlib.parent
    test_whether_aug_already = len(list(parent_dir.glob(f'{train_vid_pathlib.name}_augmented*'))) >= 10
    if test_whether_aug_already:
        continue
    for j in range(NUM_AUGMENTATIONS_PER_VID):
        augmented_train_vid_path = parent_dir / f'{train_vid_pathlib.name}_augmented_{j}.avi'
        augment_video(train_vid_pathlib, augmented_train_vid_path)

for i in constants.val_indices_of_first_5_classes:
    dict_query_result = VAL_INSTANCE_2_CROPPED.get(str(i), None)
    if dict_query_result is None:
        continue
    val_vid_pathlib = pathlib.Path(dict_query_result)
    parent_dir = val_vid_pathlib.parent
    test_whether_aug_already = len(list(parent_dir.glob(f'{val_vid_pathlib.name}_augmented*'))) >= 10
    if test_whether_aug_already:
        continue
    for j in range(NUM_AUGMENTATIONS_PER_VID):
        augmented_val_vid_path = parent_dir / f'{val_vid_pathlib.name}_augmented_{j}.avi'
        augment_video(val_vid_pathlib, augmented_val_vid_path)
        print(f"Created file {augmented_val_vid_path.resolve()}")

for i in constants.test_indices_of_first_5_classes:
    dict_query_result = TEST_INSTANCE_2_CROPPED.get(str(i), None)
    if dict_query_result is None:
        continue
    test_vid_pathlib = pathlib.Path(dict_query_result)
    parent_dir = test_vid_pathlib.parent
    test_whether_aug_already = len(list(parent_dir.glob(f'{test_vid_pathlib.name}_augmented*'))) >= 10
    if test_whether_aug_already:
        continue
    for j in range(NUM_AUGMENTATIONS_PER_VID):
        augmented_test_vid_path = parent_dir / f'{test_vid_pathlib.name}_augmented_{j}.avi'
        augment_video(test_vid_pathlib, augmented_test_vid_path)