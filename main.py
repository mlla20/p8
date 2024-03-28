import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset
import torch.optim as optim
import torchvision.transforms
import speechbrain
import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import timedelta
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


start = time.time()
batchsize = 512
# Load data
# Link til dataset doc: https://pytorch.org/audio/main/generated/torchaudio.datasets.LIBRISPEECH.html#torchaudio.datasets.LIBRISPEECH
data =  torchaudio.datasets.LIBRISPEECH(root ='./LibreSpeech', url = 'dev-clean', download= True) 

# Gets just the waveform from the data, that is all we need as both data, and label. 
data_waveform_raw = [sample[0] for sample in data]

# Normalizing all waveforms to the interval -1 to 1
def normalize_tensor(tensor):
    # Find the maximum and minimum values in the tensor
    max_val = tensor.max()
    min_val = tensor.min()
    
    # Normalize the tensor
    normalized_tensor = 2 * (tensor - min_val) / (max_val - min_val) - 1
    
    return normalized_tensor.unsqueeze(0)

waveform_norm = []

for tensor in data_waveform_raw:
    waveform_norm.extend(normalize_tensor(tensor))

# Splitting all tensors up into 
def split_tensor(tensor, split_length=160):
    tensor_length = tensor.size(1)
    num_splits = tensor_length // split_length
    split_tensors = []
    for i in range(num_splits):
        split_tensors.append(tensor[:, i * split_length : (i + 1) * split_length])
    return split_tensors

waveform = []

for tensor in waveform_norm:
    waveform.extend(split_tensor(tensor))

# Split dataset into training, validation and testsets 
waveform_train, waveform_val, waveform_test = torch.utils.data.random_split(waveform, [int(0.7*len(waveform)+1),int(0.2*len(waveform)),int(0.1*len(waveform))])
# Setting up a data loader to manage batchsize and so on.
data_loader = torch.utils.data.DataLoader(dataset=waveform_train, batch_size= batchsize, shuffle= True,)
dataiter = iter(data_loader)
waweform = next(dataiter)

# Defining the device, so the model is trained with a cuda device if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model 
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            # Remember to change input -and outputsize when changing the splicing of the data
            nn.Conv1d(1,1,7,1),
            nn.RReLU(),
            nn.Linear(154, 128),
            nn.RReLU(),
            nn.Linear(128, 64),
            nn.RReLU(),
            nn.Conv1d(1,1,2,1),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(1,1,2,1),
            nn.RReLU(),
            nn.Linear(62, 128),
            nn.RReLU(),
            nn.Linear(128, 160),
            nn.RReLU(),
            nn.Conv1d(1,1,7,1, padding=3),
            nn.Tanh()
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Define loss and optimizer 
model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Train model
epochs = 100
epochs_plot = []
output = []
train_loss_plot = []
val_loss_plot = []
print('Beginning training')
for epoch in range(epochs):
    i = 0
    for wave in data_loader:
        datas = wave.to(device)
        recon = model(datas)
        loss = criterion(recon, datas)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #if i % 2000 == 1999:
        #   print(f'Epoch: {epoch + 1}, Training loss: {loss.item():.6f}, and iteration: {i}')
        #i += 1
    # Calculating validation loss once per epoch 
    val_data = waveform_val[np.random.randint(0,len(waveform_val))].to(device)
    val_recon = model(val_data)
    val_loss = criterion(val_recon, val_data)

    train_loss_plot.append(loss.item())
    val_loss_plot.append(val_loss.item())
    epochs_plot.append(epoch + 1)

    print(f'Epoch: {epoch + 1}, Training loss: {loss.item():.6f}, and Validation loss: {val_loss.item():.6f}')

stop = time.time()
print(f'Total time training {epochs} epochs, with batchsize {batchsize}: {timedelta(seconds=(stop-start))}')

plt.figure()
plt.title('Training and Validation loss')
plt.plot(epochs_plot, train_loss_plot, color = 'blue', label = 'Training loss')
plt.plot(epochs_plot, val_loss_plot, color = 'red', label = 'Validation loss')
plt.legend()
plt.show()