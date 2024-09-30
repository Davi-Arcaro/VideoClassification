#Following https://keras.io/examples/vision/video_classification/

import os
import keras
from imutils import paths
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import imageio
import cv2
from IPython.display import Image

#Setup
img_size = 224
batch_size = 64
epochs = 10

#Hyperparameters
max_seq_length = 20
num_features = 2048

#Data prep
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv('test.csv')

print(f'Total videos for training: {len(train_df)}')
print(f'Total videos for testing: {len(test_df)}')

train_df.sample(10)

#One of many challenges of training video classifiers is figuring out a way to feed the videos to a network. In this example, we'll do the following:
#1. Capture the frames of a video
#2. Extract frames from the videos until a maximum frame count is reached
#3. In the case, where a video's frame count is lesses than the maximum frame count we will pad the video with zeros

#The following two methods are taken from this tutorial:
#https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub

def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]

def load_video(path, max_frames=0, resize=(img_size, img_size)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
                
    finally:
        cap.release()
    return np.array(frames)


#Using a pre-trained model on the ImageNet-1k dataset. Using InceptionV3 model for this purpose

def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights='imagenet',
        include_top=False,
        pooling='avg',
        input_shape=(img_size, img_size, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((img_size, img_size, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name='feature_extractor')

feature_extractor = build_feature_extractor()


#Labels of the video are strings. Using StringLookup to convert strings to a numerical form for the neural network to understand.

label_processor = keras.layers.StringLookup(
    num_oov_indices=0, vocabulary=np.unique(train_df['tag'])
)
print(label_processor.get_vocabulary())


#Putting all the pieces together to create the data processing utility

def prepare_all_videos(df, root_dir):
    num_samples = len(df)
    video_paths = df['video_name'].values.tolist()
    labels = df['tag'].values
    labels = keras.ops.convert_to_numpy(label_processor(labels[..., None]))

    #frame_masks and frame_features are what we will feed to our sequence model.
    #frame_masks will contain a bunch of booleans denoting if a timestep is masked with padding or not.

    frame_masks = np.zeros(shape=(num_samples, max_seq_length), dtype='bool') 
    frame_features = np.zeros(shape=(num_samples, max_seq_length, num_features), dtype='float32')

    #For each video
    for idx, path in enumerate(video_paths):
        #Gather all its frames and add a batch dimension.
        frames = load_video(os.path.join(root_dir, path))
        frames = frames[None, ...]

        #Initialize placeholders to store the masks and features of the current video.
        temp_frame_mask = np.zeros(shape=(1, max_seq_length,), dtype='bool',)
        temp_frame_features = np.zeros(shape=(1, max_seq_length, num_features),dtype='float32')

        #Extract features from the frames of the current video.
        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            length = min(max_seq_length, video_length)
            for j in range(length):
                temp_frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :], verbose=0,)
                temp_frame_mask[i, :length] = 1 #1 = not masked, 0 = masked
        
        frame_features[idx,] = temp_frame_features.squeeze()
        frame_masks[idx,] = temp_frame_mask.squeeze()
    
    return(frame_features, frame_masks), labels

train_data, train_labels = prepare_all_videos(train_df, 'train')
test_data, test_labels = prepare_all_videos(test_df, 'test')

print(f'Frame features in train set: {train_data[0].shape}')
print(f'Frame masks in train set: {train_data[1].shape}')