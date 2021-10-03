import torch
import torch.nn as nn
import torch.nn.functional as fun
import numpy as np
import torchvision.utils as vutils
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from vqvae import VQVAE


def train(loader, model, opt, writer, device = torch.device('cuda'), beta = 1.0, steps = 0):
    
    loop = tqdm(loader, leave = True, position = 0)
    model.train()
    for imgs, _ in loop:
        
        imgs = imgs.to(device)
        opt.zero_grad()
        z_e_x, z_q_x, x_tilde = model(imgs)
        
        recon_loss = fun.mse_loss(x_tilde, imgs)
        vq_loss = fun.mse_loss(z_q_x, z_e_x.detach())
        commitment_loss = fun.mse_loss(z_e_x, z_q_x.detach())
        
        loss = recon_loss + vq_loss + beta * commitment_loss
        loss.backward()
        
        writer.add_scalar("loss/train/reconstruction", recon_loss.item(), steps)
        writer.add_scalar("loss/train/quantization", vq_loss.item(), steps)
        writer.add_scalar("loss/train/commitment", commitment_loss.item(), steps)
        
        opt.step()
        steps += 1
        
def test(loader, model, writer, device = torch.device('cuda'), steps = 0):
    
    loop = tqdm(loader, leave = True, position = 0)
    model.eval()
    with torch.no_grad():
        recon_loss, vq_loss, commitment_loss = 0.0, 0.0, 0.0
        for imgs, _ in loop:
            
            imgs = imgs.to(device)
            z_e_x, z_q_x, x_tilde = model(imgs)
            recon_loss += fun.mse_loss(x_tilde, imgs)
            vq_loss += fun.mse_loss(z_q_x, z_e_x)
            commitment_loss += fun.mse_loss(z_e_x, z_q_x)
        
        recon_loss /= len(loader)
        vq_loss /= len(loader)
        commitment_loss /= len(loader)
        
    writer.add_scalar("loss/train/reconstruction", recon_loss.item(), steps)
    writer.add_scalar("loss/train/quantization", vq_loss.item(), steps)
    writer.add_scalar("loss/train/commitment", commitment_loss.item(), steps)
    
    return recon_loss.item(), vq_loss.item(), commitment_loss.item()

def generate(model, imgs, device = torch.device('cuda')):
    
    model.eval()
    with torch.no_grad():
        
        imgs = imgs.to(device)
        _, _, x_tilde = model(imgs)
    
    return x_tilde

def main():
    
    TRAIN_DIR = '/content/train'
    TEST_DIR = '/content/test'
    MODEL_DIR = '/content'
    
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                                    ])
    
    train_dataset = datasets.CIFAR10(root = TRAIN_DIR , train = True, transform = transform, download = True)
    test_dataset = datasets.CIFAR10(root = TEST_DIR, train = False, transform = transform, download = True)
    train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True, num_workers = 2)
    test_loader = DataLoader(test_dataset, batch_size= 64)
    device = torch.device('cuda')
    model = VQVAE(3, 256).to(device)
    opt = torch.optim.Adam(model.parameters(), lr = 3e-4)
    
    writer = SummaryWriter('logs/VqVae')
    test_images, _ = next(iter(test_loader))
    test_grid = vutils.make_grid(test_images, nrows = 8, range = (-1,1), normalize = True)
    writer.add_image('target', test_grid, 0)
    recon_imgs = generate(model, test_images)
    grid = vutils.make_grid(recon_imgs.cpu(), nrows = 8, range = (-1,1), normalize = True)
    writer.add_image('Reconstructed', grid, 0)
    
    best_loss = np.Inf
    
    for epoch in range(0, 100):
        
        train(train_loader, model, opt, writer)
        loss, _, _ = test(test_loader, model, writer)
        
        recon_imgs = generate(model, test_images)
        grid = vutils.make_grid(recon_imgs.cpu(), nrows = 8, range = (-1,1), normalize = True)
        writer.add_image('Reconstructed', grid, epoch + 1)
        
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), open(MODEL_DIR + 'Vqvae_model.pth', 'wb'))
            
if __name__ == '__main__':
    
    main()