# Cell 1
import os
import numpy as np
from PIL import Image

import torch
print(f'PyTorch version: {torch.__version__}')

from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import CenterCrop, ToTensor


# Cell 2
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
    
print(f'Using device: {device}')


# Cell 3
class Kitti2015StereoDataset(Dataset):
    def __init__(self, training):
        if training == True:
            data_path = 'data/kitti_2015_stereo_example/training'
            print('Loading training dataset on RAM')
        else:
            data_path = 'data/kitti_2015_stereo_example/validation'
            print('Loading test dataset on RAM')

        left_image_file_list    = os.listdir(f'{data_path}/left_image')
        right_image_file_list   = os.listdir(f'{data_path}/right_image')
        disparity_file_list     = os.listdir(f'{data_path}/disparity')

        image_to_frame_transform = transforms.Compose(
            [CenterCrop((256, 512)),
             ToTensor()]
        )
        
        self.left_image_memory = []
        for file_name in left_image_file_list:
            image   = Image.open(f'{data_path}/left_image/{file_name}')
            tensor  = image_to_frame_transform(image)
            self.left_image_memory.append(tensor)
        
        self.right_image_memory = []
        for file_name in right_image_file_list:
            image   = Image.open(f'{data_path}/right_image/{file_name}')
            tensor  = image_to_frame_transform(image)
            self.right_image_memory.append(tensor)
     
        self.disparity_memory = []
        self.valid_mask_memory = []
        for file_name in disparity_file_list:
            array   = np.asarray(Image.open(f'{data_path}/disparity/{file_name}')) / 256.0

            center_crop = CenterCrop((256, 512))
            tensor = center_crop(
                torch.Tensor(array).
                unsqueeze(0)
            )
            self.disparity_memory.append(tensor)

            valid_mask = tensor.bool().float()
            self.valid_mask_memory.append(valid_mask)

    def __len__(self):
        return len(self.left_image_memory)
    
    def __getitem__(self, index):
        left_image_memory   = self.left_image_memory[index]
        right_image_memory  = self.right_image_memory[index]
        disparity_memory    = self.disparity_memory[index]
        valid_mask_memory   = self.valid_mask_memory[index]
    
        return left_image_memory, right_image_memory, disparity_memory, valid_mask_memory

print('-------------------------------')        
training_dataset    = Kitti2015StereoDataset(training = True)
test_dataset        = Kitti2015StereoDataset(training = False)

mini_batch_size         = 16
training_data_loader    = DataLoader(training_dataset, batch_size = mini_batch_size, shuffle = True)
test_data_loader        = DataLoader(test_dataset, batch_size = mini_batch_size)

print('-------------------------------')
print(f'Training dataset size:              {len(training_dataset)}')
print(f'Number of training mini-batches:    {len(training_data_loader)}')
print('')
print(f'Test dataset size:                  {len(test_dataset)}')
print(f'Number of test mini-batches:        {len(test_data_loader)}')
for x_left, x_right, t, _ in training_data_loader:
    print('-------------------------------')
    print(f'Shape of left input:    {x_left.shape} {x_left.dtype}')
    print(f'Shape of right input:   {x_right.shape} {x_right.dtype}')
    print(f'Shape of target:        {t.shape} {t.dtype}')
    print('-------------------------------')
    break


# Cell 4
class NeuralNetwork(nn.Module):
    def parameter_initializer(self, layer):
        if hasattr(layer, 'weight') and hasattr(layer, 'bias'):
            torch.nn.init.xavier_normal_(layer.weight) 
            torch.nn.init.zeros_(layer.bias)

    def __init__(self):
        super().__init__()

        self.cnn_stack1 = nn.Sequential(
            nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5, stride = 2, padding = 2, padding_mode = 'zeros'),
            nn.GELU(),            
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 5, stride = 2, padding = 2, padding_mode = 'zeros'),
            nn.GELU()
        )

        self.cnn_stack2 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5, stride = 2, padding = 2, padding_mode = 'zeros'),
            nn.GELU(),
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 5, stride = 2, padding = 2, padding_mode = 'zeros'),
            nn.GELU(),
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 5, stride = 2, padding = 2, padding_mode = 'zeros'),
            nn.GELU()
        )

        self.decnn_stack1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size = 2, stride = 2),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 2, stride = 2),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels = 64, out_channels = 32, kernel_size = 2, stride = 2),
            nn.GELU(),
        )

        self.decnn_stack2 = nn.Sequential(            
            nn.ConvTranspose2d(in_channels = 32, out_channels = 8, kernel_size = 2, stride = 2),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels = 8, out_channels = 1, kernel_size = 2, stride = 2)
        )

        for module in self.modules():		
            module.apply(self.parameter_initializer)

    def forward(self, x_left, x_right):
        x = torch.concat(
            [x_left, x_right], 
            dim = 1
        )
        
        cnn1    = self.cnn_stack1(x)
        cnn2    = self.cnn_stack2(cnn1)

        decnn1  = self.decnn_stack1(cnn2)
        decnn2  = self.decnn_stack2(decnn1 + cnn1)

        if not self.training:
            decnn2 = torch.clamp(decnn2, 0.0, 256.0)

        return decnn2
    
model = NeuralNetwork().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3, betas = (0.9, 0.999), weight_decay = 1e-4)


# Cell 5
def train(device, data_loader, model, optimizer):
    total_loss = 0

    model.train()
    for x_left, x_right, t, mask in data_loader:
        x_left  = x_left.to(device)
        x_right = x_right.to(device)
        
        y = model(x_left, x_right)
        t = t.to(y.device)

        mask    = mask.to(device)
        y       = y * mask
        loss    = torch.nn.functional.mse_loss(y, t)
                
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        mini_batch_size = x_left.shape[0]
        total_loss += loss.item() * mini_batch_size
    
    dataset_size = len(data_loader.dataset)
    average_loss = total_loss / dataset_size

    return average_loss

def test(device, data_loader, model):
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for x_left, x_right, t, mask in data_loader:
            x_left  = x_left.to(device)
            x_right = x_right.to(device)

            y = model(x_left, x_right)
            t = t.to(y.device)
            
            mask    = mask.to(device)
            y       = y * mask
            loss    = torch.nn.functional.mse_loss(y, t)

            mini_batch_size = x_left.shape[0]
            total_loss += loss.item() * mini_batch_size
    
    dataset_size = len(data_loader.dataset)
    average_loss = total_loss / dataset_size

    return average_loss

def inference(device, x_left, x_right, model):
    x_left  = x_left.view(1, 3, 256, 512)
    x_right = x_right.view(1, 3, 256, 512)
    
    model.eval()
    with torch.no_grad():
        x_left  = x_left.to(device)
        x_right = x_right.to(device)

        y = model(x_left, x_right)
        y = y.view(256, 512)

    return y


# Cell 6
max_epoch = 500
for t in range(max_epoch):
    if t % 50 == 0 or t == max_epoch - 1:
        print(f'Epoch {t + 1 :>3d} / {max_epoch}')

    training_loss = train(device, training_data_loader, model, optimizer)
    if t % 50 == 0 or t == max_epoch - 1:
        print('  Training progress')
        print(f'    Average loss:   {training_loss :>8f}')

    if t % 50 == 0 or t == max_epoch - 1:
        test_loss = test(device, test_data_loader, model)
        print('  Validation performance')
        print(f'    Average loss:   {test_loss :>8f}')

        if t + 1 < max_epoch:
            print()
    
torch.save(model.state_dict(), 'model.pth')
print('-------------------------------')
print('Saved PyTorch ANN parameters to model.pth')
print('-------------------------------')


# Cell 7
import random
import matplotlib.pyplot as plt

model = NeuralNetwork().to(device)
model.load_state_dict(torch.load('model.pth', weights_only = True))

test_sample_index = random.randint(0, len(test_dataset) - 1)
x_left, x_right, t, _ = test_dataset[test_sample_index]
y = inference(device, x_left, x_right, model)

tensor_to_image = transforms.ToPILImage()
left_image      = tensor_to_image(x_left)
right_image     = tensor_to_image(x_right)
target_image    = t.view(256, 512)
output_image    = y.to('cpu')

figure, axes = plt.subplots(4, 1, figsize = (20, 30))
axes[0].imshow(left_image)
axes[1].imshow(right_image)
axes[2].imshow(target_image, cmap = 'gray')
axes[3].imshow(output_image, cmap = 'gray')
plt.savefig('result.png')
plt.show()