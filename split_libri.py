import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
from vector_quantize_pytorch import ResidualVQ
from torch.utils.data import Dataset
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

file_path = "tensors_waveform_train_clean_100_split160.pt"
batchsize = 1024
# Load data
# Link til dataset doc: https://pytorch.org/audio/main/generated/torchaudio.datasets.LIBRISPEECH.html#torchaudio.datasets.LIBRISPEECH
data =  torchaudio.datasets.LIBRISPEECH(root = '/home/student.aau.dk/mlla20/p8/LibriSpeech', url = 'train-clean-100', download= True) 

# Defining the device, so the model is trained with a cuda device if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Normalizing data')
# Gets just the waveform from the data, that is all we need as both data, and label. 
data_waveform_raw = [sample[0] for sample in data]

# Normalizing all waveforms to the interval -1 to 1
def normalize_tensor(tensor):
    # Find the maximum and minimum values in the tensor
    max_val = tensor.max()
    min_val = tensor.min()
    
    # Normalize the tensor in-place
    tensor.sub_(min_val).div_(max_val - min_val).mul_(2).sub_(1)
    
    # Reshape tensor to have an additional dimension
    return tensor.unsqueeze(0)

waveform_norm = []

for tensor in data_waveform_raw:
    waveform_norm.extend(normalize_tensor(tensor))

print('Splitting data')
# Splitting all tensors up into 
def split_tensor(tensor, split_length=160):
    tensor_length = tensor.size(1)
    num_splits = tensor_length // split_length
    
    for i in range(num_splits):
        yield tensor[:, i * split_length : (i + 1) * split_length]

waveform = []

for tensor in waveform_norm:
    waveform.extend(split_tensor(tensor))

torch.save(waveform, file_path)