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

batchsize = 2048
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
def split_tensor(tensor, split_length=320):
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

class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1)

    def forward(self, x):
        return self._conv_forward(F.pad(x, [self.causal_padding, 0]), self.weight, self.bias)


class CausalConvTranspose1d(nn.ConvTranspose1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1) + self.output_padding[0] + 1 - self.stride[0]
    
    def forward(self, x, output_size=None):
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose1d')

        assert isinstance(self.padding, tuple)
        output_padding = self._output_padding(
            x, output_size, self.stride, self.padding, self.kernel_size, self.dilation)
        return F.conv_transpose1d(
            x, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)[...,:-self.causal_padding]


class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        
        self.dilation = dilation

        self.layers = nn.Sequential(
            CausalConv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=7, dilation=dilation),
            nn.ELU(),
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=1)
        )

    def forward(self, x):
        return self.layers(x)


class EncoderBlock(nn.Module):
    def __init__(self, out_channels, stride):
        super().__init__()

        self.layers = nn.Sequential(
            ResidualUnit(in_channels=out_channels//2,
                         out_channels=out_channels//2, dilation=1),
            nn.ELU(),
            ResidualUnit(in_channels=out_channels//2,
                         out_channels=out_channels//2, dilation=3),
            nn.ELU(),
            ResidualUnit(in_channels=out_channels//2,
                         out_channels=out_channels//2, dilation=9),
            nn.ELU(),
            CausalConv1d(in_channels=out_channels//2, out_channels=out_channels,
                      kernel_size=2*stride, stride=stride)
        )

    def forward(self, x):
        return self.layers(x)


class DecoderBlock(nn.Module):
    def __init__(self, out_channels, stride):
        super().__init__()

        self.layers = nn.Sequential(
            CausalConvTranspose1d(in_channels=2*out_channels,
                               out_channels=out_channels,
                               kernel_size=2*stride, stride=stride),
            nn.ELU(),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                         dilation=1),
            nn.ELU(),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                         dilation=3),
            nn.ELU(),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                         dilation=9),

        )

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(self, C, D):
        super().__init__()

        self.layers = nn.Sequential(
            CausalConv1d(in_channels=1, out_channels=C, kernel_size=7),
            nn.ELU(),
            EncoderBlock(out_channels=2*C, stride=2),
            nn.ELU(),
            EncoderBlock(out_channels=4*C, stride=2),
            nn.ELU(),
            EncoderBlock(out_channels=8*C, stride=2),
            nn.ELU(),
            #EncoderBlock(out_channels=16*C, stride=2),
            #nn.ELU(),
            CausalConv1d(in_channels=8*C, out_channels=D, kernel_size=3)
        )

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self, C, D):
        super().__init__()
        
        self.layers = nn.Sequential(
            CausalConv1d(in_channels=D, out_channels=8*C, kernel_size=7),
            nn.ELU(),
            #DecoderBlock(out_channels=8*C, stride=2),
            #nn.ELU(),
            DecoderBlock(out_channels=4*C, stride=2),
            nn.ELU(),
            DecoderBlock(out_channels=2*C, stride=2),
            nn.ELU(),
            DecoderBlock(out_channels=C, stride=2),
            nn.ELU(),
            CausalConv1d(in_channels=C, out_channels=1, kernel_size=7)
        )
    
    def forward(self, x):
        return self.layers(x)


class SoundStream(nn.Module):
    def __init__(self, C, D):
        super().__init__()

        self.encoder = Encoder(C=C, D=D)
        #self.quantizer = ResidualVQ(
        #   num_quantizers=n_q, dim=D, codebook_size=codebook_size,
        #   kmeans_init=True, kmeans_iters=100, threshold_ema_dead_code=2
        #)
        self.decoder = Decoder(C=C, D=D)
    
    def forward(self, x):
        #print("Input tensor shape:", x.shape)
        e = self.encoder(x)
        #print("Encoder output shape:", e.shape)
        #quantized, _, _ = self.quantizer(e)
        #print("Quantized tensor shape:", quantized.shape)
        o = self.decoder(e)
        #print("Decoder output shape:", o.shape)
        return o
    
model = SoundStream(1, 1).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

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

print(f'Total time training {epochs} epochs, with batchsize {batchsize}')

plt.figure()
plt.title('Training and Validation loss')
plt.plot(epochs_plot, train_loss_plot, color = 'blue', label = 'Training loss')
plt.plot(epochs_plot, val_loss_plot, color = 'red', label = 'Validation loss')
plt.legend()
plt.savefig('/home/student.aau.dk/mlla20/p8/first_SS.png')