import pathlib
from tensorflow import keras
import numpy as np

from frame_generator import frames_from_video_file
from optical_flow_generator import compute_video_optical_flow
from constants import CLASSES_TB_TRAINED_UPON

MODEL_obj_map = { 'K': keras.backend }

OPTFLOW_MODEL_DIR = pathlib.Path('my_saved_optFlow_model_iter2')
optFlow_model = keras.models.load_model(OPTFLOW_MODEL_DIR, custom_objects=MODEL_obj_map)

RGB_MODEL_DIR = pathlib.Path('my_saved_RGB_model_iter2')
RGB_model = keras.models.load_model(RGB_MODEL_DIR, custom_objects=MODEL_obj_map)

# since keras stores labels in a sorted alphabetical order and since we our model is not sequential we don't have any way accessing the labels
# We therefore hard-code them all the way
LABELS = sorted(CLASSES_TB_TRAINED_UPON) 

example_vid_path = pathlib.Path('data') / 'teacher.avi'

RGB_model_input = frames_from_video_file(example_vid_path)
RGB_model_input = np.expand_dims(RGB_model_input, axis=0)


RGB_prediction = RGB_model.predict(RGB_model_input)[0] # because we only have one input in our batch
print(list(zip(LABELS, RGB_prediction)))

video_frames = frames_from_video_file(example_vid_path)
optFlow_model_input = compute_video_optical_flow(video_frames)
optFlow_model_input = np.expand_dims(optFlow_model_input, axis=0)


optFlow_prediction = optFlow_model.predict(optFlow_model_input)[0]
print(list(zip(LABELS, optFlow_prediction)))

aggregated_preds = [(RGB_prediction[i] + optFlow_prediction[i] * 2) / 3 
                    for i, _ in enumerate(RGB_prediction)]

print(list(zip(LABELS, aggregated_preds)))