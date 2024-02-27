import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, models
import torch.optim as optim
import os
import random
import cv2
import numpy as np

class_label_real = 0
class_label_attack = 1

#Replay Attack
data_path_train_real_RA = '/home/taha/FASdatasets/Sohail/Replay_Attack/train/real'
data_path_train_attack_fixed_RA = '/home/taha/FASdatasets/Sohail/Replay_Attack/train/attack/fixed'
data_path_train_attack_hand_RA = '/home/taha/FASdatasets/Sohail/Replay_Attack/train/attack/hand'

data_path_devel_real_RA = '/home/taha/FASdatasets/Sohail/Replay_Attack/devel/real'
data_path_devel_attack_fixed_RA = '/home/taha/FASdatasets/Sohail/Replay_Attack/devel/attack/fixed'
data_path_devel_attack_hand_RA = '/home/taha/FASdatasets/Sohail/Replay_Attack/devel/attack/hand'

# Replay Mobile Real
data_path_train_real_RM = '/home/taha/FASdatasets/Sohail/Replay-Mobile/train/real'
data_path_train_attack_RM = '/home/taha/FASdatasets/Sohail/Replay-Mobile/train/attack'

data_path_devel_real_RM = '/home/taha/FASdatasets/Sohail/Replay-Mobile/devel/real'
data_path_devel_attack_RM = '/home/taha/FASdatasets/Sohail/Replay-Mobile/devel/attack'



def load_samples(path, class_label, transform): #Select N frames returned from read_all_frames and assign labels to all samples of same class
    frames = read_all_frames(path)
    total_frames = list(range(0, frames.shape[0], 1))
    selected_samples = random.sample(total_frames, 1)
    samples =[]
    # Assign the same class label to all samples
    label = class_label
    samples =(transform(frames[selected_samples].squeeze()), label)     
    return samples

def read_all_frames(video_path): # reads all frames from a particular video and converts them to PyTorch tensors.
    frame_list = []
    video = cv2.VideoCapture(video_path)
    success = True
    while success:
        success, frame = video.read()
        if success:
            frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA) #framesize kept 40, 30 as mentioned in paper but 224, 224 is also fine 
            frame_list.append(frame)
    frame_list = np.array(frame_list)
    return frame_list

class VideoDataset(Dataset):
    def __init__(self, data_path, class_label):
        self.data_path = data_path #path for directory containing video files
        self.video_files = [file for file in os.listdir(data_path) if file.endswith('.mov') or file.endswith('.mp4')] #list of video files in the specified directory #.mov for RA and RM, .mp4 for RY
        self.class_label = class_label #manually assign class_label for your desired class while loading
        self.data_length = len(self.video_files) 
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self): # returns the total number of samples in the dataset
        return self.data_length

    def __getitem__(self, idx): # loads and returns a sample from the dataset at the given index
        file = self.video_files[idx]
        path = os.path.join(self.data_path, file)
        frames= load_samples(path, self.class_label, self.transform)

        return frames
    
# Replay Attack Dataset

train_dataset_real_RA = VideoDataset(data_path_train_real_RA, class_label_real)
train_dataset_attack_fixed_RA = VideoDataset(data_path_train_attack_fixed_RA, class_label_attack)
train_dataset_attack_hand_RA = VideoDataset(data_path_train_attack_hand_RA, class_label_attack)

val_dataset_real_RA = VideoDataset(data_path_devel_real_RA, class_label_real)
val_dataset_attack_fixed_RA = VideoDataset(data_path_devel_attack_fixed_RA, class_label_attack)
val_dataset_attack_hand_RA = VideoDataset(data_path_devel_attack_hand_RA, class_label_attack)

# Replay Mobile Dataset

train_dataset_real_RM = VideoDataset(data_path_train_real_RM, class_label_real)
train_dataset_attack_RM = VideoDataset(data_path_train_attack_RM, class_label_attack)

val_dataset_real_RM = VideoDataset(data_path_devel_real_RM, class_label_real)
val_dataset_attack_RM = VideoDataset(data_path_devel_attack_RM, class_label_attack)


# Replay Attack DataLoader

train_loader_real_RA = DataLoader(train_dataset_real_RA, batch_size=1, shuffle=True)
train_loader_attack_fixed_RA = DataLoader(train_dataset_attack_fixed_RA, batch_size=1, shuffle=True)
train_loader_attack_hand_RA = DataLoader(train_dataset_attack_hand_RA, batch_size=1, shuffle=True)

val_loader_real_RA = DataLoader(val_dataset_real_RA, batch_size=1, shuffle=False)
val_loader_attack_fixed_RA = DataLoader(val_dataset_attack_fixed_RA, batch_size=1, shuffle=False)
val_loader_attack_hand_RA = DataLoader(val_dataset_attack_hand_RA, batch_size=1, shuffle=False)

# Replay Mobile DataLoader

train_loader_real_RM = DataLoader(train_dataset_real_RM, batch_size=1, shuffle=True)
train_loader_attack_RM = DataLoader(train_dataset_attack_RM, batch_size=1, shuffle=True)

val_loader_real_RM = DataLoader(val_dataset_real_RM, batch_size=1, shuffle=False)
val_loader_attack_RM = DataLoader(val_dataset_attack_RM, batch_size=1, shuffle=False)

# Concatenate Replay Attack + Replay Mobile Dataset

concatenated_train_dataset_RA_RM = ConcatDataset([train_dataset_real_RA, train_dataset_attack_fixed_RA, train_dataset_attack_hand_RA, train_dataset_real_RM, train_dataset_attack_RM])
concatenated_val_dataset_RA_RM = ConcatDataset([val_dataset_real_RA, val_dataset_attack_fixed_RA, val_dataset_attack_hand_RA, val_dataset_real_RM, val_dataset_attack_RM])

concatenated_train_loader = DataLoader(concatenated_train_dataset_RA_RM, batch_size=64, shuffle=True, pin_memory=True, num_workers=8)
concatenated_val_loader = DataLoader(concatenated_val_dataset_RA_RM, batch_size=64, shuffle=False, pin_memory=True, num_workers=8)

# Print dataset sizes
print(f"Training set size: {len(concatenated_train_dataset_RA_RM)}")
print(f"Validation set size: {len(concatenated_val_dataset_RA_RM)}")


# Load pre-trained ResNet18
# model = models.resnet18(pretrained=True)
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, 2)

# Load pre-trained MobileNetV2
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(in_features=1280, out_features=2) #default in_features =1280, out_features = 1000
# print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 50

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

train_losses = []
val_losses = []

train_accuracies = []
val_accuracies = []

# Set up early stopping parameters
patience = 5  # Number of epochs with no improvement after which training will be stopped
best_loss = float('inf') #set to positive infinity to ensure that the first validation loss encountered will always be considered an improvement
counter = 0  # Counter to keep track of consecutive epochs with no improvement

#Training loop
for epoch in range(epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    train_correct_predictions = 0
    total_train_samples = 0

    for train_images, train_labels in concatenated_train_loader:
        train_images, train_labels = train_images.to(device), train_labels.to(device)
        # Clear the gradients
        optimizer.zero_grad()
        # Forward Pass
        train_outputs = model(train_images)
        # Find the Loss
        train_loss = criterion(train_outputs, train_labels)
        # Calculate gradients
        train_loss.backward()
        # Update Weights
        optimizer.step()

        # accumulate the training loss
        running_loss += train_loss.item()

        # calculate training accuracy
        _, train_predicted = torch.max(train_outputs, 1) # _ contain max value, train_predicted contain the indices where maximum value occured
        train_correct_predictions += (train_predicted == train_labels).sum().item() 
        total_train_samples += train_labels.size(0)
            
    train_total_loss = running_loss / len(concatenated_train_loader)
    train_accuracy = train_correct_predictions / total_train_samples * 100
    train_losses.append(train_total_loss)
    train_accuracies.append(train_accuracy)

    val_running_loss = 0.0
    val_correct_prediction = 0
    total_val_samples = 0

    #Validation
    model.eval()
    with torch.no_grad():
        for val_images, val_labels in concatenated_val_loader:
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            val_op = model(val_images)
            val_loss = criterion(val_op, val_labels)
            val_running_loss += val_loss.item()

            _, val_predicted = torch.max(val_op, 1)
            val_correct_prediction += (val_predicted == val_labels).sum().item()
            total_val_samples += val_labels.size(0)
        
        val_total_loss = val_running_loss / len(concatenated_val_loader)
        val_accuracy = val_correct_prediction / total_val_samples * 100
        val_losses.append(val_total_loss)
        val_accuracies.append(val_accuracy)

    # Check if validation loss has improved
    if val_total_loss < best_loss:
        best_loss = val_total_loss
        counter = 0
        # Save the model if needed
        torch.save(model.state_dict(), 'RA_RM_best_model.pth')

    else:
        counter += 1

        # Check if training should be stopped
        if counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            break

    print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {train_total_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%, Validation Loss: {val_total_loss: .4f}, Best Loss: {best_loss: .4f}, Validation Accuracy: {val_accuracy:.2f}%')


