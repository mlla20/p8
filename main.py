import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset
import torch.optim as optim
import torchvision.transforms
import speechbrain
import matplotlib.pyplot as plt

# Load data
# Link til dataset doc: https://pytorch.org/audio/main/generated/torchaudio.datasets.LIBRISPEECH.html#torchaudio.datasets.LIBRISPEECH
data =  torchaudio.datasets.LIBRISPEECH(root ='./LibreSpeech', url = 'dev-clean', download= True) 

# Gets just the waveform from the data, that is all we need as both data, and label. 
data_waveform_raw = [sample[0] for sample in data]

#Splitting all tensors up into 
def split_tensor(tensor, split_length=512):
    tensor_length = tensor.size(1)
    num_splits = tensor_length // split_length
    split_tensors = []
    for i in range(num_splits):
        split_tensors.append(tensor[:, i * split_length : (i + 1) * split_length])
    return split_tensors

waveform = []

for tensor in data_waveform_raw:
    waveform.extend(split_tensor(tensor))


# Setting up a data loader to manage batchsize and so on.
#print(len(waveform)) # number of samples
data_loader = torch.utils.data.DataLoader(dataset=waveform, batch_size= 16, shuffle= True,)

dataiter = iter(data_loader)
waweform = next(dataiter)
#print(torch.min(waveform[0]), torch.max(waveform[0]))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Define model 

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            # Remember to change input -and outputsize when changing the splicing of the data
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
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
epochs = 1
output = []
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
        if i % 2000 == 1999:
           print(f'Epoch: {epoch + 1}, loss: {loss.item():.6f}, and {i}')
        i += 1
