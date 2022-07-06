import torch
from byol_pytorch import BYOL
from torchvision import models
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

device == "cuda"
round_num = 0

train_dataset = CIFAR10(
    root='dataset',
    train=True,
    #transform=TransformsSimCLR(size=image_size),
    download=True
)  # 训练数据集

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=20,
    drop_last=True,
)  # 训练数据加载器

iter_trainloader = iter(trainloader)

def get_next_train_batch():
    try:
        # Samples a new batch for persionalizing
        (x, y) = next(iter_trainloader)
    except StopIteration:
        # restart the generator if the previous generator is exhausted.
        iter_trainloader = iter(trainloader)
        (x, y) = next(iter_trainloader)

    if type(x) == type([]):
        x[0] = x[0].to(device)
    else:
        x = x.to(device)
    y = y.to(device)

    return x, y

resnet = models.resnet18(pretrained=True)

renet_device = resnet.to(device)

learner = BYOL(
    renet_device,
    image_size = 32,
    hidden_layer = 'avgpool'
)

opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

def sample_unlabelled_images():
    return torch.randn(20, 3, 32, 32)

for _ in range(100):
    #images = sample_unlabelled_images()
    print('epoch: ',round_num,' begin')
    images, y = get_next_train_batch()
    print('loaded training data')    
    loss = learner(images)
    print('training loss got')        
    opt.zero_grad()
    loss.backward()
    print('backward finished')    
    opt.step()
    learner.update_moving_average() # update moving average of target encoder
    print('---------------------------')    

# save your improved network
torch.save(resnet.state_dict(), './improved-net.pt')