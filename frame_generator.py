import pathlib
import random
import cv2
import tensorflow as tf
import numpy as np


def format_frames(frame, output_size):
    frame = tf.image.convert_image_dtype(frame, tf.float32)
    frame = tf.image.resize_with_pad(frame, *output_size)
    return frame


def frames_from_video_file(video_path, n_frames, output_size=(224, 224), frame_step=15):
    '''
    '''
    result = []
    src = cv2.VideoCapture(str(video_path))

    video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)

    need_length = 1 + (n_frames - 1) * frame_step

    if need_length > video_length:
        start = 0
    else:
        max_start = video_length - need_length
        start = random.randint(0, max_start+1)

    src.set(cv2.CAP_PROP_POS_FRAMES, start)

    ret, frame = src.read()
    result.append(format_frames(frame, output_size))

    for _ in range(n_frames - 1):
        for _ in range(frame_step):
            ret, frame = src.read()
        if ret:
            frame = format_frames(frame, output_size)
            result.append(frame)
        else:
            result.append(np.zeros_like(result[0]))
    src.release()
    result = np.array(result)[..., [2, 1, 0]]

    return result


class FrameGenerator:
    def __init__(self, path: pathlib.Path, videos_dir: pathlib.Path, idx_2_label: dict, training: bool = False, instance_idx: list = [], n_frames: int = 74):
        '''
        Reads frames from the given (root) path, 
        and returns them coupled with their associated label/class
        The root directory containslabel folders and each of these contains folders for a given instance of that label/class, which has the extracted 
        frames of that instance's video clip 

        Args:
            path: root path where frames are stored into subdirectories
            training: determine whether the training dataset is being generated
        '''
        self.path = path
        self.videos_dir = videos_dir
        self.idx_2_label = idx_2_label
        self.training = training
        self.instance_idx = instance_idx
        self.n_frames = n_frames
        self.class_names = sorted(
            set(p.name for p in self.path.iterdir() if p.is_dir()))
        self.class_ids = dict((name, idx)
                              for idx, name in enumerate(self.class_names))

    def get_frames_and_class_name_for_each_instance(self):
        result = list()
        for idx in self.instance_idx:
            frames_dir_str = f'instance_{idx}'
            frames_paths = list(self.path.glob(
                '*/' + frames_dir_str + '/*.png'))
            if len(frames_paths) == 0:
                print(f'No frames have been extracted for instance {idx}')
                continue
            label = frames_paths[0].parent.parent.name
            result.append((frames_paths, label))
        return result

    def get_videos_files_and_class_names(self):
        video_file_paths = []
        classes = []
        for idx in self.instance_idx:
            video_file_paths.extend(self.videos_dir.glob(f'{idx}.mp4'))
            classes.append(self.idx_2_label[idx]['clean_text'])
        return video_file_paths, classes

    def __call__(self):
        video_paths, classes = self.get_videos_files_and_class_names()
        pairs = list(zip(video_paths, classes))

        if self.training:
            random.shuffle(pairs)

        for path, name in pairs:
            video_frames = frames_from_video_file(path, self.n_frames)
            label = self.class_ids[name] # encode labels
            yield video_frames, label


# NUM_FRAMES = 74

# data_dir_path = pathlib.Path('data')
# output_dir_path = data_dir_path / 'output'
# frames_dir_path = output_dir_path / 'frames'
# videos_dir_path = data_dir_path / 'videos'
# cropped_videos_dir_path = videos_dir_path / 'cropped'
# dataset_dir_path = data_dir_path / 'dataset'
# cleansed_training_data_path = pathlib.Path('cleansed_train_ds.json')

# with open(cleansed_training_data_path, 'r') as f_:
#     training_data = json.load(f_)
#     f_.close()

# instance_idx = [45, 46, 59, 121, 122, 178, 179, 180, 181, 184, 199, 200, 275, 276, 277, 278, 279, 292, 316, 328, 329, 330, 334, 340, 367,
#                 396, 421, 470, 477, 596, 633, 634, 635, 636, 637, 644, 686, 700, 775, 885, 995, 1084, 1149, 1241, 1310, 1421, 1424, 1661, 1967, 2037]
# fg = FrameGenerator(frames_dir_path, cropped_videos_dir_path,
#                     training_data, True, instance_idx)
# i = 0
# for video_frames, label in fg():
#     print(i, video_frames.shape, label)
#     i += 1