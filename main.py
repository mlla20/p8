import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

batchsize = 4096
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

# Split dataset into training, validation and testsets 
waveform_train, waveform_val, waveform_test = torch.utils.data.random_split(waveform, [int(0.7*len(waveform)),int(0.2*len(waveform)),int(0.1*len(waveform))])
# Setting up a data loader to manage batchsize and so on.
data_loader = torch.utils.data.DataLoader(dataset=waveform_train, batch_size= batchsize, shuffle= True,)
dataiter = iter(data_loader)
waweform = next(dataiter)

# Define model 
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            # Remember to change input -and outputsize when changing the splicing of the data
            nn.Conv1d(1,4,7,1, padding = 'same'),
            nn.RReLU(),
            # Dilation kernals
            nn.Conv1d(4,8,3,1, dilation= 1, padding = 'same'),
            nn.Conv1d(8,8,1,1, padding = 'same'),
            nn.Conv1d(8,8,3,1, dilation= 3, padding = 'same'),
            nn.Conv1d(8,8,1,1, padding = 'same'),
            nn.Conv1d(8,8,3,1, dilation= 9, padding = 'same'),
            nn.Conv1d(8,8,1,1, padding = 'same'),
            # Dilation kernals
            nn.Conv1d(8,8,5,1, dilation= 1, padding = 'same'),
            nn.Conv1d(8,8,1,1, padding = 'same'),
            nn.Conv1d(8,8,5,1, dilation= 3, padding = 'same'),
            nn.Conv1d(8,8,1,1, padding = 'same'),
            nn.Conv1d(8,8,5,1, dilation= 9, padding = 'same'),
            nn.Conv1d(8,8,1,1, padding = 'same'),
            # Dilation kernals
            nn.Conv1d(8,8,7,1, dilation= 1, padding = 'same'),
            nn.Conv1d(8,8,1,1, padding = 'same'),
            nn.Conv1d(8,8,7,1, dilation= 3, padding = 'same'),
            nn.Conv1d(8,8,1,1, padding = 'same'),
            nn.Conv1d(8,8,7,1, dilation= 9, padding = 'same'),
            nn.Conv1d(8,8,1,1, padding = 'same'),
            # Convolution to reduce dimension
            nn.Conv1d(8,16,4,2, padding= 2),
            nn.RReLU(),
            # Dilation kernals
            nn.Conv1d(16,16,3,1, dilation= 1, padding = 'same'),
            nn.Conv1d(16,16,1,1, padding = 'same'),
            nn.Conv1d(16,16,3,1, dilation= 3, padding = 'same'),
            nn.Conv1d(16,16,1,1, padding = 'same'),
            nn.Conv1d(16,16,3,1, dilation= 9, padding = 'same'),
            nn.Conv1d(16,16,1,1, padding = 'same'),
            # Dilation kernals
            nn.Conv1d(16,16,5,1, dilation= 1, padding = 'same'),
            nn.Conv1d(16,16,1,1, padding = 'same'),
            nn.Conv1d(16,16,5,1, dilation= 3, padding = 'same'),
            nn.Conv1d(16,16,1,1, padding = 'same'),
            nn.Conv1d(16,16,5,1, dilation= 9, padding = 'same'),
            nn.Conv1d(16,16,1,1, padding = 'same'),
            # Dilation kernals
            nn.Conv1d(16,16,7,1, dilation= 1, padding = 'same'),
            nn.Conv1d(16,16,1,1, padding = 'same'),
            nn.Conv1d(16,16,7,1, dilation= 3, padding = 'same'),
            nn.Conv1d(16,16,1,1, padding = 'same'),
            nn.Conv1d(16,16,7,1, dilation= 9, padding = 'same'),
            nn.Conv1d(16,16,1,1, padding = 'same'),
            # Convolution to reduce dimension
            nn.Conv1d(16,16,4,2),
            nn.RReLU(),
            # Dilation kernals
            nn.Conv1d(16,16,3,1, dilation= 1, padding = 'same'),
            nn.Conv1d(16,16,1,1, padding = 'same'),
            nn.Conv1d(16,16,3,1, dilation= 3, padding = 'same'),
            nn.Conv1d(16,16,1,1, padding = 'same'),
            nn.Conv1d(16,16,3,1, dilation= 9, padding = 'same'),
            nn.Conv1d(16,16,1,1, padding = 'same'),
            # Dilation kernals
            nn.Conv1d(16,16,5,1, dilation= 1, padding = 'same'),
            nn.Conv1d(16,16,1,1, padding = 'same'),
            nn.Conv1d(16,16,5,1, dilation= 3, padding = 'same'),
            nn.Conv1d(16,16,1,1, padding = 'same'),
            nn.Conv1d(16,16,5,1, dilation= 9, padding = 'same'),
            nn.Conv1d(16,16,1,1, padding = 'same'),
            # Dilation kernals
            nn.Conv1d(16,16,7,1, dilation= 1, padding = 'same'),
            nn.Conv1d(16,16,1,1, padding = 'same'),
            nn.Conv1d(16,16,7,1, dilation= 3, padding = 'same'),
            nn.Conv1d(16,16,1,1, padding = 'same'),
            nn.Conv1d(16,16,7,1, dilation= 9, padding = 'same'),
            nn.Conv1d(16,16,1,1, padding = 'same'),
            # Convolution to reduce dimension
            nn.Conv1d(16,16,4,2),
            nn.RReLU(),
            nn.Conv1d(16,8,3,1, padding='same'),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(8,16,3,1, padding= 'same'),
            nn.RReLU(),
            # Transpose convoluiton to upsample 
            nn.ConvTranspose1d(16,16,3,2, padding= 0, output_padding= 1),
            # Dilation kernals
            nn.Conv1d(16,16,3,1, dilation= 1,padding = 'same'),
            nn.Conv1d(16,16,1,1, padding = 'same'),
            nn.Conv1d(16,16,3,1, dilation= 3,padding = 'same'),
            nn.Conv1d(16,16,1,1, padding = 'same'),
            nn.Conv1d(16,16,3,1, dilation= 9,padding = 'same'),
            nn.Conv1d(16,16,1,1, padding = 'same'),
            # Dilation kernals
            nn.Conv1d(16,16,5,1, dilation= 1,padding = 'same'),
            nn.Conv1d(16,16,1,1, padding = 'same'),
            nn.Conv1d(16,16,5,1, dilation= 3,padding = 'same'),
            nn.Conv1d(16,16,1,1, padding = 'same'),
            nn.Conv1d(16,16,5,1, dilation= 9,padding = 'same'),
            nn.Conv1d(16,16,1,1, padding = 'same'),
            # Dilation kernals
            nn.Conv1d(16,16,7,1, dilation= 1,padding = 'same'),
            nn.Conv1d(16,16,1,1, padding = 'same'),
            nn.Conv1d(16,16,7,1, dilation= 3,padding = 'same'),
            nn.Conv1d(16,16,1,1, padding = 'same'),
            nn.Conv1d(16,16,7,1, dilation= 9,padding = 'same'),
            nn.Conv1d(16,16,1,1, padding = 'same'),
            nn.RReLU(),
            # Transpose convolution to upsample
            nn.ConvTranspose1d(16,16,3,2, padding= 0, output_padding= 1),
            # Dilation kernals
            nn.Conv1d(16,16,3,1, dilation= 1,padding = 'same'),
            nn.Conv1d(16,16,1,1, padding = 'same'),
            nn.Conv1d(16,16,3,1, dilation= 3,padding = 'same'),
            nn.Conv1d(16,16,1,1, padding = 'same'),
            nn.Conv1d(16,16,3,1, dilation= 9,padding = 'same'),
            nn.Conv1d(16,16,1,1, padding = 'same'),
            # Dilation kernals
            nn.Conv1d(16,16,5,1, dilation= 1,padding = 'same'),
            nn.Conv1d(16,16,1,1, padding = 'same'),
            nn.Conv1d(16,16,5,1, dilation= 3,padding = 'same'),
            nn.Conv1d(16,16,1,1, padding = 'same'),
            nn.Conv1d(16,16,5,1, dilation= 9,padding = 'same'),
            nn.Conv1d(16,16,1,1, padding = 'same'),
            # Dilation kernals
            nn.Conv1d(16,16,7,1, dilation= 1,padding = 'same'),
            nn.Conv1d(16,16,1,1, padding = 'same'),
            nn.Conv1d(16,16,7,1, dilation= 3,padding = 'same'),
            nn.Conv1d(16,16,1,1, padding = 'same'),
            nn.Conv1d(16,16,7,1, dilation= 9,padding = 'same'),
            nn.Conv1d(16,16,1,1, padding = 'same'),
            nn.RReLU(),
            # Transpose convoluiton to upsample
            nn.ConvTranspose1d(16,8,5,2, padding= 0, output_padding= 1),
            # Dilation kernals
            nn.Conv1d(8,8,3,1, dilation= 1,padding = 'same'),
            nn.Conv1d(8,8,1,1, padding = 'same'),
            nn.Conv1d(8,8,3,1, dilation= 3,padding = 'same'),
            nn.Conv1d(8,8,1,1, padding = 'same'),
            nn.Conv1d(8,8,3,1, dilation= 9,padding = 'same'),
            nn.Conv1d(8,8,1,1, padding = 'same'),
            # Dilation kernals
            nn.Conv1d(8,8,5,1, dilation= 1,padding = 'same'),
            nn.Conv1d(8,8,1,1, padding = 'same'),
            nn.Conv1d(8,8,5,1, dilation= 3,padding = 'same'),
            nn.Conv1d(8,8,1,1, padding = 'same'),
            nn.Conv1d(8,8,5,1, dilation= 9,padding = 'same'),
            nn.Conv1d(8,8,1,1, padding = 'same'),
            # Dilation kernals
            nn.Conv1d(8,8,7,1, dilation= 1,padding = 'same'),
            nn.Conv1d(8,8,1,1, padding = 'same'),
            nn.Conv1d(8,8,7,1, dilation= 3,padding = 'same'),
            nn.Conv1d(8,8,1,1, padding = 'same'),
            nn.Conv1d(8,8,7,1, dilation= 9,padding = 'same'),
            nn.Conv1d(8,4,1,1, padding = 'same'),
            nn.RReLU(),
            nn.Conv1d(4,1,5,1, padding='same'),
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

#print(sum(p.numel() for p in model.parameters()))
#print(sum(p.numel() for p in model.parameters() if p.requires_grad))

# Train model
epochs = 150
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

    # Calculating validation loss once per epoch 
    val_data = waveform_val[np.random.randint(0,len(waveform_val))].to(device)
    val_recon = model(val_data)
    val_loss = criterion(val_recon, val_data)

    train_loss_plot.append(loss.item())
    val_loss_plot.append(val_loss.item())
    epochs_plot.append(epoch + 1)

    print(f'Epoch: {epoch + 1}, Training loss: {loss.item():.6f}, and Validation loss: {val_loss.item():.6f}')

plt.figure()
plt.title('Training and Validation loss')
plt.plot(epochs_plot, train_loss_plot, color = 'blue', label = 'Training loss')
plt.plot(epochs_plot, val_loss_plot, color = 'red', label = 'Validation loss')
plt.legend()
plt.savefig('/home/student.aau.dk/mlla20/p8/ungabungaplot.png')