import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image

resnet = models.resnet50()
# modules = list(resnet.children())[:-1]
# resnet = nn.Sequential(*modules)
resnet.eval()

def infer_resnet(model, file_path):
    vector_img = np.load(file_path)
    tensor_data = torch.tensor(vector_img)
    tensor_data = tensor_data.permute(0, 3, 1, 2)/255.0

    with torch.no_grad():
        feature = model(tensor_data)
    reshaped_feature = feature.view(feature.shape[0], feature.shape[1])
    return reshaped_feature

root_path = "/home/kaisa/Desktop/Toan/violence-detection/dataset/frames_stack"
violence_path = root_path + "/Violence"
non_violence_path = root_path + "/NonViolence"

violence_files = [violence_path + "/" + x for x in os.listdir(violence_path)]
non_violence_files = [non_violence_path + "/" + x for x in os.listdir(non_violence_path)]

violence_features = []
non_violence_features = []

num_of_frames = 15

frames_extract_path = "/home/kaisa/Desktop/Toan/violence-detection/dataset/frames_extract"
if not os.path.exists(frames_extract_path):
    os.makedirs(frames_extract_path)

frames_extract_violence_path = frames_extract_path + "/Violence"
if not os.path.exists(frames_extract_violence_path):
    os.makedirs(frames_extract_violence_path)

for file in tqdm(violence_files):
    feature = infer_resnet(resnet, file)
    if len(feature) < num_of_frames:
        continue
    path_save = frames_extract_violence_path + "/" + file.split("/")[-1].split(".")[0] + ".pt"
    torch.save(feature, path_save)


frames_extract_non_violence_path = frames_extract_path + "/NonViolence"
if not os.path.exists(frames_extract_non_violence_path):
    os.makedirs(frames_extract_non_violence_path)

for file in tqdm(non_violence_files):
    feature = infer_resnet(resnet, file)
    if len(feature) < num_of_frames:
        continue
    print(len(feature))
    path_save = frames_extract_non_violence_path + "/" + file.split("/")[-1].split(".")[0] + ".pt"
    torch.save(feature, path_save)
