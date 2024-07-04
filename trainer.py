import os
import wandb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim

import torch.nn as nn
import torch.nn.functional as F

from model import LSTMAttention, LSTMWithMultiHeadAttention, MLP


wandb.login()
run = wandb.init(
    # Set the project where this run will be logged
    project="violence-detection",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": 0.001,
        "epochs": 300,
    },
)

class CustomImageDataset(Dataset):
    def __init__(self, labels, transform=None):
        self.transform = transform
        self.labels = labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        file_path = self.labels[idx]
        frames = torch.load(file_path)
        label_file = file_path.split('/')[-2]
        if label_file == 'Violence':
            label = 1
        else:
            label = 0
        if self.transform:
            frames = self.transform(frames)
        return frames, label
    


list_labels = [os.path.join('dataset/frames_extract', label, x) for label in os.listdir('dataset/frames_extract') for x in os.listdir(os.path.join('dataset/frames_extract', label))]
train_split, test_split = train_test_split(list_labels, test_size=0.2)

train_dataset = CustomImageDataset(train_split)
test_dataset = CustomImageDataset(test_split)


train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=True)




model = LSTMAttention(1000, 512, 2)
model = model.to('cuda')
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


num_epochs = 300
min_loss = 100

max_f1_score = 0
max_precision = 0
max_recall = 0
max_accuracy = 0
losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    for frames, labels in train_dataloader:
        frames = frames.to('cuda')
        # Forward pa
        outputs = model(frames)
        one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=2).float()
        one_hot_labels = one_hot_labels.to('cuda')

        loss = criterion(outputs, one_hot_labels)
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        avg_loss = running_loss/len(train_dataloader)
    losses.append(avg_loss)

    if avg_loss < min_loss:
        min_loss = avg_loss
        torch.save(model, 'models/model.pth')

    #eval

    preds = []
    trues = []
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for frames, labels in test_dataloader:
            frames = frames.to('cuda')
            labels = labels.to('cuda')
            outputs = model(frames)
            _, predicted = torch.max(outputs.data, 1)
            preds.extend(predicted.cpu().tolist())
            trues.extend(labels.cpu().tolist())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        # print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
        # trues = [int(x) for x in trues]
        # preds = [int(x) for x in preds]
        accuracy = accuracy_score(trues, preds)
        precision = precision_score(trues, preds)
        recall = recall_score(trues, preds)
        f1 = f1_score(trues, preds)
        wandb.log({"accuracy": accuracy, "loss": avg_loss, "precision": precision, "recall": recall, "f1": f1, "epoch": epoch})


        if f1 > max_f1_score:
            print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}')
            max_accuracy = accuracy
            max_precision = precision
            max_recall = recall
            max_f1_score = f1
            torch.save(model, 'models/model.pth')


print(max_accuracy, max_precision, max_recall, max_f1_score)






# model = MLP(input_size=5 * 1000, num_classes=2)
# model = model.to('cuda')
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0001)


# num_epochs = 100
# min_loss = 100
# for epoch in range(num_epochs):
#     running_loss = 0
#     for frames, labels in train_dataloader:
#         frames = frames.to('cuda')

#         # Forward pa
#         outputs = model(frames)
#         one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=2).float()
#         one_hot_labels = one_hot_labels.to('cuda')
#         loss = criterion(outputs, one_hot_labels)
#         # Backward pass and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#     if running_loss/len(train_dataloader) < min_loss:
#         min_loss = running_loss
#         torch.save(model, 'models/model.pth')

#     #eval
#     model.eval()
#     with torch.no_grad():
#         correct = 0
#         total = 0
#         for frames, labels in test_dataloader:
#             frames = frames.to('cuda')
#             labels = labels.to('cuda')
#             outputs = model(frames)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#         print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))




# classs_output = 2
# LSTM_multihead = LSTMWithMultiHeadAttention(1000, 512, 12, 8, 2)
# LSTM_multihead = LSTM_multihead.to('cuda')


# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(LSTM_multihead.parameters(), lr=0.0001, momentum=0.9)

# num_epochs = 100
# min_loss = 100
# for epochs in range(num_epochs):
#     running_loss = 0.0
#     LSTM_multihead.train()
#     for frames, labels in train_dataloader:

#         optimizer.zero_grad()
#         frames = frames.to('cuda')
#         outputs = LSTM_multihead(frames)
#         print(outputs[0])

#         one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=2).float()

    
#         # labels = convert_to_one_hot(labels)
#         one_hot_labels = one_hot_labels.to('cuda')
#         loss = criterion(outputs[0] , one_hot_labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#         print(loss)
#     if running_loss/len(train_dataloader) < min_loss:
#         min_loss = running_loss
#         torch.save(LSTM_multihead.state_dict(), 'models/model.pth')
