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

batchsize = 1024
file_path = "p8/tensors_gaus_40.pt"

# Defining the device, so the model is trained with a cuda device if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading the data from the generated file
waveform = torch.load(file_path)

# Split dataset into training, validation and testsets 
waveform_train, waveform_val = torch.utils.data.random_split(waveform, [int(0.5*len(waveform))+(len(waveform)-2*int(0.5*len(waveform))),int(0.5*len(waveform))])

# Setting up a data loader to manage batchsize and so on.
data_loader = torch.utils.data.DataLoader(dataset=waveform_train, batch_size= batchsize, shuffle= True,)
dataiter = iter(data_loader)
waweform = next(dataiter)

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
           nn.Linear(40, 40),
           nn.ELU(),
           nn.Linear(40, 40),
           nn.ELU(),
           nn.Linear(40, 40),
           nn.ELU(),
           nn.Linear(40, 40),
           nn.ELU(),
           nn.Linear(40, 40),
           nn.ELU(),
           nn.Linear(40, 40),
           nn.ELU(),
           nn.Linear(40, 40),
           nn.ELU(),
           nn.Linear(40, 10),
           nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(10, 40),
            nn.ELU(),
            nn.Linear(40, 40),
            nn.ELU(),
            nn.Linear(40, 40),
            nn.ELU(),
            nn.Linear(40, 40),
            nn.ELU(),
            nn.Linear(40, 40),
            nn.ELU(),
            nn.Linear(40, 40),
            nn.ELU(),
            nn.Linear(40, 40),
            nn.ELU(),
            nn.Linear(40, 40),
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
epochs = 300
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
plt.savefig('/home/student.aau.dk/mlla20/p8/plots/gaus_40.png')

