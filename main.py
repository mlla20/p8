import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset
import torchvision.transforms
import speechbrain
import matplotlib.pyplot as plt

# Load data
# Link til dataset doc: https://pytorch.org/audio/main/generated/torchaudio.datasets.LIBRISPEECH.html#torchaudio.datasets.LIBRISPEECH
data =  torchaudio.datasets.LIBRISPEECH(root ='./LibreSpeech', url = 'dev-clean', download= True) 
#print(data[1][0].shape)


# Define model 
# Define loss and optimizer 

# Train model
lr = 0.01
epochs = 1000

#for epoch in range(epochs):
    # Compute prediction

    # Compute Loss
    # Computer Gradient 
    # Update weights 