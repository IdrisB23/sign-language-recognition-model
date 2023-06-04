from vidaug import augmentors as va
import cv2
import numpy as np
import ffmpeg
import pathlib
from constants import MODEL_INPUT_IMG_SIZE

def sample_frames(vid_path: pathlib.Path):
    cap = cv2.VideoCapture(str(vid_path.resolve()))
    sampled_frames = []
    curr_frame = 0
    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            sampled_frames.append(frame)
            curr_frame += 1
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    return np.array(sampled_frames)

def sometimes(aug): return va.Sometimes(0.5, aug)

def augment_video(input_path: pathlib.Path, output_path: pathlib.Path):

    sampled_frames = sample_frames(input_path)

    # Begin: 0-pad all frames in videos if their frames are smaller than the cropping size

    frame_height = sampled_frames.shape[1]
    frame_width = sampled_frames.shape[2]
    nb_pads_vertically = 0
    nb_pads_above = 0
    nb_pads_below = 0
    if frame_height < MODEL_INPUT_IMG_SIZE[0]:
        nb_pads_vertically = MODEL_INPUT_IMG_SIZE[0] - frame_height
        nb_pads_above = nb_pads_vertically // 2
        nb_pads_below = nb_pads_vertically - nb_pads_above
    
    nb_pads_horizontally = 0
    nb_pads_left = 0
    nb_pads_right = 0
    if frame_width < MODEL_INPUT_IMG_SIZE[1]:
        nb_pads_horizontally = MODEL_INPUT_IMG_SIZE[1] - frame_width
        nb_pads_left = nb_pads_horizontally // 2
        nb_pads_right = nb_pads_horizontally - nb_pads_left

    sampled_frames = np.pad(sampled_frames, ((0,0), (nb_pads_above, nb_pads_below), (nb_pads_left, nb_pads_right), (0, 0)), 'constant', constant_values=0)

    # End: 0-pad all frames in videos if their frames are smaller than the cropping size

    seq = va.Sequential([
        #va.RandomCrop(size=MODEL_INPUT_IMG_SIZE),
        va.RandomRotate(degrees=10),
        sometimes(va.HorizontalFlip())
    ])
    video_aug = seq(sampled_frames)
    video_aug = np.array(video_aug, dtype=np.float64)
    video_aug_denorm = video_aug * 255
    video_aug_denorm = video_aug_denorm.astype(np.uint8)
    _, height, width, _ = video_aug_denorm.shape

    process = (
        ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
            .output(str(output_path.resolve()), pix_fmt='yuv420p', vcodec='libx264', r='60')
            .overwrite_output()
            .run_async(pipe_stdin=True)
    )
    for frame in video_aug_denorm:
        process.stdin.write(
            frame
                .astype(np.uint8)
                .tobytes()
        )
    process.stdin.close()
    process.wait()


def main():
    pass


if __name__ == '__main__':
    main()