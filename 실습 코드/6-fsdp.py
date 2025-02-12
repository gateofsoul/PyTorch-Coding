# Cell 1
import random
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import torchvision
from torchvision.transforms import ToTensor


# Cell 2
# -------- Excution time and VRAM usage estimation -------- #
start_event = torch.cuda.Event(enable_timing = True)
end_event = torch.cuda.Event(enable_timing = True)

def initialize_cuda_performace_record():
    torch.cuda.init()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    start_event.record() 
    
def get_cuda_performace_record():
    end_event.record()
    
    torch.cuda.synchronize()
    excution_time = start_event.elapsed_time(end_event) / 1000
    peak_VRAM_usage = torch.cuda.max_memory_allocated()

    return excution_time, peak_VRAM_usage


# Cell 3
# -------- Dataset and data loader -------- #
def get_dataset(train, download):
    dataset = torchvision.datasets.FashionMNIST(root = 'data', train = train, download = download, transform = ToTensor())

    return dataset

def get_data_loader(distributed, dataset, mini_batch_size, shuffle): 
    if distributed:
        data_loader = DataLoader(dataset, batch_size = mini_batch_size, pin_memory = True, sampler = DistributedSampler(dataset, shuffle = shuffle))
    else:
        data_loader = DataLoader(dataset, batch_size = mini_batch_size, pin_memory = True, shuffle = shuffle)

    return data_loader


# Cell 4
# -------- ANN architecture ------ #
class NeuralNetwork(nn.Module):
    def weight_initializer(self, layer):
        if hasattr(layer, 'weight') and hasattr(layer, 'bias'):
            torch.nn.init.xavier_normal_(layer.weight) 
            torch.nn.init.zeros_(layer.bias)
            
    def __init__(self):
        super().__init__()
        
        self.cnn_stack1 = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 4, kernel_size = 5, stride = 2, padding = 2, padding_mode = 'zeros'),
            nn.GELU(),
            nn.Conv2d(in_channels = 4, out_channels = 8, kernel_size = 5, stride = 2, padding = 1, padding_mode = 'zeros'),
            nn.GELU(),
            nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 5, stride = 2, padding = 1, padding_mode = 'zeros'),
            nn.GELU()
        )

        self.cnn_stack2 = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 4, kernel_size = 5, stride = 2, padding = 2, padding_mode = 'zeros'),
            nn.GELU(),
            nn.Conv2d(in_channels = 4, out_channels = 8, kernel_size = 5, stride = 2, padding = 2, padding_mode = 'zeros'),
            nn.GELU()
        )
        
        self.linear_stack1 = nn.Sequential(
            nn.Flatten(start_dim = 1),
            nn.Linear(16 * 2 * 2, 32)
        )
        
        self.linear_stack2 = nn.Sequential(
            nn.Flatten(start_dim = 1),
            nn.Linear(8 * 7 * 7, 128),
            nn.Tanh(),
            nn.Linear(128, 32)
        )

        self.softmax_stack = nn.Sequential(
            nn.Linear(32, 10),
            nn.Softmax(dim = 1)
        )
        
        for module in self.modules():		
            module.apply(self.parameter_initializer)

    @autocast(device_type = 'cuda')
    def forward(self, x):
        x1 = self.cnn_stack1(x)
        x1 = self.linear_stack1(x1)

        x2 = self.cnn_stack2(x)
        x2 = self.linear_stack2(x2)

        x = x1 + x2
        y = self.softmax_stack(x)
        
        return y
    

# Cell 5
# -------- Saving and loading checkpoints -------- #
def save_checkpoint(ddp_model, optimizer, scheduler):
    checkpoint = {
        'model_state'       : ddp_model.module.state_dict(),
        'optimizer_state'   : optimizer.state_dict(),
        'scheduler_state'   : scheduler.state_dict()
    }

    torch.save(checkpoint, 'checkpoint.pth')

def load_checkpoint(ddp_model, optimizer, scheduler):
    checkpoint = torch.load('checkpoint.pth', weights_only = False)

    ddp_model.module.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    scheduler.load_state_dict(checkpoint['scheduler_state'])

# -------- ANN Training -------- #
def train(data_loader, ddp_model, optimizer, accumulation_number = 1):
    local_rank = int(os.environ['LOCAL_RANK']) 		
    distributed_loss = torch.zeros(2).to(local_rank)
    
    ddp_model.train()
    scaler = GradScaler()
    for mini_batch_index, (x, t) in enumerate(data_loader):
        x = x.to(local_rank)
        
        y = ddp_model(x)
        t = t.to(y.device)
        loss = torch.nn.functional.cross_entropy(y, t)

        scaler.scale(loss).backward()

        if mini_batch_index % accumulation_number == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1e-1)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        mini_batch_size = x.shape[0]
        distributed_loss[0] += loss.item() * mini_batch_size
        distributed_loss[1] += mini_batch_size
    dist.all_reduce(distributed_loss, op = dist.ReduceOp.SUM)
    
    average_loss = distributed_loss[0] / distributed_loss[1]
    
    return average_loss

def training_loop(dataset, mini_batch_size, max_epoch, checkpoint_interval, accumulation_number = 1):
    world_size  = int(os.environ['WORLD_SIZE'])
    global_rank = int(os.environ['RANK'])
    local_rank  = int(os.environ['LOCAL_RANK'])
    
    if local_rank == 0:
        initialize_cuda_performace_record()

    training_data_loader = get_data_loader(distributed = True, dataset = dataset, mini_batch_size = mini_batch_size, shuffle = True)

    model = NeuralNetwork().to(local_rank)
    ddp_model = DDP(model)
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr = 1e-2, betas = (0.9, 0.999), weight_decay = 1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.5)

    current_epoch = 0
    if os.path.exists('checkpoint.pth'):
        load_checkpoint(ddp_model, optimizer, scheduler)
        current_epoch = scheduler.last_epoch
        
        if local_rank == 0:
            print(f'Resuming training from checkpoint at epoch {current_epoch + 1}\n' +
                  '\n', 
                  end = ''
            )
        dist.barrier()

    for t in range(current_epoch, max_epoch):
        training_data_loader.sampler.set_epoch(t)
        print(f'Worker {global_rank + 1} / {world_size} begins Epoch {t + 1 :> 3d} / {max_epoch}\n', end = '')
        training_loss = train(training_data_loader, ddp_model, optimizer, accumulation_number)
        scheduler.step()
        
        if local_rank == 0:
            print(f'  Training average loss: {training_loss :>8f}\n', end = '')
                
            if t + 1 < max_epoch:
                print('\n', end = '')        
        dist.barrier()

        if (t + 1) % checkpoint_interval == 0 and (t + 1) != max_epoch:
            if local_rank == 0:
                save_checkpoint(ddp_model, optimizer, scheduler)
                print(f'Saved training checkpoint at {t + 1} epochs to "checkpoint.pth"\n' +
                        '\n', 
                        end = ''
                )        
        dist.barrier()

    if global_rank == 0:
        excution_time, peak_VRAM_usage = get_cuda_performace_record()
        print('-------------------------------\n' +
              f'Training with DDP for {max_epoch} epochs:\n' +
              f'  Execution time:   {excution_time :>0.4} sec\n' +
              f'  Peak VRAM usage:  {peak_VRAM_usage / (1024 ** 2) :>,.2f} MB\n' +
              '-------------------------------\n',
              end = ''
        )

        torch.save(ddp_model.module.state_dict(), 'model.pth')
        print('Saved PyTorch ANN parameters to model.pth\n' +
              '-------------------------------\n',
              end = ''
        )


# Cell 6
# -------- ANN test and inference -------- #            
def test(device, data_loader, model):
    total_loss = 0
    total_correct = 0
    
    model.eval()
    with torch.no_grad():
        for x, t in data_loader:
            x = x.to(device)

            y = model(x)
            t = t.to(y.device)
            loss = torch.nn.functional.cross_entropy(y, t)

            mini_batch_size = x.shape[0]
            total_loss += loss.item() * mini_batch_size
            total_correct += (y.argmax(dim = 1) == t).type(torch.float).sum().item()

    dataset_size = len(data_loader.dataset)
    average_loss = total_loss / dataset_size
    accuracy = total_correct / dataset_size

    return average_loss, accuracy

def inference(device, data, model):
    x = data.view(1, 1, 28, 28)

    model.eval()
    with torch.no_grad():
        x = x.to(device)
        y = model(x)
    
    return y


# Cell 7
# -------- Main function ------ #
if __name__ == '__main__':
    number_of_GPU       = torch.cuda.device_count()
    world_size          = int(os.environ['WORLD_SIZE'])
    local_world_size    = int(os.environ['LOCAL_WORLD_SIZE'])
    global_rank         = int(os.environ['RANK'])
    local_rank          = int(os.environ['LOCAL_RANK'])

    if local_rank == 0:
        print(f'PyTorch version: {torch.__version__}\n' +
              '-------------------------------\n' +
              f'Number of GPU:      {number_of_GPU}\n' +
              f'World size:         {world_size}\n' +
              f'Local world size:   {local_world_size}\n' +
              '-------------------------------\n', 
              end = ''
        )

    if local_world_size > number_of_GPU:
        if local_rank == 0:
            print(f'Need more GPUs in this node\n' +
                  f'  Number of GPU in this node:   {number_of_GPU}\n' +
                  f'  This node needs:              {local_world_size}\n' +
                  '-------------------------------\n', 
                  end = '')
        exit()

    # ---- Training ---- #
    dist.init_process_group(backend = 'nccl')

    if local_rank == 0:
        training_dataset = get_dataset(train = True, download = True)
    dist.barrier()
    
    if local_rank != 0:
        training_dataset = get_dataset(train = True, download = False)

    training_mini_batch_size    = 64
    max_epoch                   = 20
    accumulation_number         = 4
    checkpoint_interval         = 5
    training_loop(training_dataset, training_mini_batch_size, max_epoch, checkpoint_interval, accumulation_number)

    dist.destroy_process_group()

    # ---- Test and inference ---- #
    if global_rank == 0:
        infernece_device        = 'cuda'
        test_dataset            = get_dataset(train = False, download = True)
        test_mini_batch_size    = 64
        test_data_loader        = get_data_loader(distributed = False, dataset = test_dataset, mini_batch_size = test_mini_batch_size, shuffle = False)

        model = NeuralNetwork().to(infernece_device)
        model.load_state_dict(torch.load('model.pth', weights_only = True))

        test_loss, test_accuracy = test(infernece_device, test_data_loader, model)
        print('Test performance\n' +
              f'  Average loss: {test_loss :>8f}\n' +
              f'  Accuracy:     {(100 * test_accuracy) :>0.2f}%\n' +
              '-------------------------------\n',
              end = ''
        )

        test_sample_index = random.randint(0, len(test_dataset) - 1)
        x, t = test_dataset[test_sample_index][0], test_dataset[test_sample_index][1]
        
        y = inference(infernece_device, x, model)

        classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        predicted, actual = classes[y.argmax(dim = 1)], classes[t]
        print('Random sample inference\n' +
              f'  Predicted: "{predicted}", Actual: "{actual}"\n' +
              '-------------------------------\n',
              end = ''
        )