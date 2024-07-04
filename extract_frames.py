import os
import cv2
from tqdm import tqdm
import numpy as np

def extract_frames(video_path, num_frames=15, target_size=(224,224)):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(10, total_frames - 1, num_frames)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Resize frame to target size
            frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB (optional)
            frames.append(frame)
            #swap frame to RGB
            # path_save = "save/frame%d.jpg" % idx
            # cv2.imwrite(path_save, frame)

        else:
            break

    cap.release()
    return frames


for fp in tqdm(os.listdir('dataset/Real Life Violence Dataset')):
    file_path = os.path.join('dataset/Real Life Violence Dataset', fp)
    new_file_path = f'dataset/frames_stack/{fp}'
    if not os.path.exists(new_file_path):
        os.makedirs(new_file_path)

    for video in os.listdir(file_path):
        video_path = os.path.join(file_path, video)
        print(video_path)
        frames = extract_frames(video_path)
        if frames is None:
            continue
        array_frames = np.array(frames)
        np.save(f'{new_file_path}/{video}.npy', array_frames)

