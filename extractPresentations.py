import argparse
import numpy as np
import torch

from solver import Solver
from utils import str2bool
from dataset import return_cts, getdata
from model import FactorVAE1, FactorVAE256
import pandas as pd
from torchvision.utils import make_grid, save_image
from utils import grid2gif, mkdirs
from torchvision import transforms
import torch.nn.functional as F
import os


def inference(model, dataset):

    namelist = []
    Z = []
    for datapoint in dataset:

        image, name = datapoint[0], datapoint[1][0]
        x_recon, mu, logvar, z = model(image)
        Z.append(z)
        namelist.append(name)
    return namelist, Z


def reconstruct(model, dataset, size=64):
    namelist = []
    for datapoint in dataset:
        image, name = datapoint[0], datapoint[1][0]
        x_recon, mu, logvar, z = model(image)
        namelist.append(name)
        x_recon = F.sigmoid(x_recon)
        x_recon = x_recon.view(1, size, size)
        print(x_recon.shape)
        pilimage = transforms.ToPILImage()(x_recon)
        name = name[name.find('.')-8:]
        path = "result256_Jan27/"+name
        pilimage.save(path)
    return image, z


def merge(pred, sample):
    result = sample
    for i in range(pred.shape[2]):
        for j in range(pred.shape[3]):
            if pred[:,:, i, j] == 0:
                result[:,:, i,j] = 0
            elif pred[:,:, i, j] != 0:
                result[:, :, i, j] = pred[:,:, i, j]
    return result


def visualize_traverse(VAE, data_loader, limit=3, inter=2/3, loc=-1, z_dim=64, output_dir='traverse_result'):

    decoder = VAE.decode
    encoder = VAE.encode
    interpolation = torch.arange(-limit, limit+0.1, inter)
    global_iter = 10

    fixed_idx = 0
    fixed_img = data_loader.dataset.__getitem__(fixed_idx)[1]
    fixed_img = fixed_img.to('cpu').unsqueeze(0)
    fixed_img_z = encoder(fixed_img)[:, :z_dim]

    # random_z = torch.rand(1, z_dim, 1, 1, device='cpu')

    Z = {'fixed_img': fixed_img_z}
    index_feature = [2, 20, 22, 26, 41]
    gifs = []
    for key in Z:
        z_ori = Z[key]
        samples = []
        for row in index_feature:
            if loc != -1 and row != loc:
                continue
            z = z_ori.clone()
            for val in interpolation:
                z[:, row, : ,:] = val
                sample = F.sigmoid(decoder(z)).data
                sample = fixed_img + sample
                # sample = merge(sample, fixed_img)
                print("fixed image shape {}".format(fixed_img.shape))
                samples.append(sample)
                gifs.append(sample)

    # samples = torch.cat(samples, dim=0).cpu()
    # title = '{}_latent_traversal(iter:{})'.format(key, 1)

    output_dir = os.path.join(output_dir, str(global_iter))
    mkdirs(output_dir)
    gifs = torch.cat(gifs)
    print("gif size is {}".format(gifs.shape))
    gifs = gifs.view(len(Z), len(index_feature), len(interpolation), 1, 256, 256).transpose(1, 2)

    for i, key in enumerate(Z.keys()):
        for j, val in enumerate(interpolation):
            save_image(tensor=gifs[i][j].cpu(),
                       filename=os.path.join(output_dir, '{}_{}.jpg'.format(key, j)),
                       nrow=z_dim, pad_value=1)

        grid2gif(str(os.path.join(output_dir, key+'*.jpg')),
                 str(os.path.join(output_dir, key+'.gif')), delay=10)

        # self.net_mode(train=True)

# def merge(pred, sample):
#     result = sample
#     for i in pred.shape[2]:
#         for j in pred.shape[3]:
#             if pred[:,:, i, j] == 0:
#                 result[:,:, i,j] = 0
#             elif  pred[:,:, i, j] != 0:
#                 result[:, :, i, j] = pred[:,:, i, j]
#     return result






def visualize_rep(VAE, data_loader, limit=3, inter=2/3, loc=-1, z_dim = 128, output_dir='traverse_result'):

    decoder = VAE.decode
    encoder = VAE.encode
    interpolation = torch.arange(-limit, limit + 0.1, inter)


    fixed_idx = 0
    fixed_img = data_loader.dataset.__getitem__(fixed_idx)
    fixed_img = fixed_img.to('cpu').unsqueeze(0)
    fixed_img_z = encoder(fixed_img)[:, :z_dim]

    random_z = torch.rand(1, z_dim, 1, 1, device='cpu')

    Z = {'fixed_img': fixed_img_z, 'random_z': random_z}

    gifs = []
    for key in Z:
        z_ori = Z[key]
        samples = []
        for row in range(z_dim):
            if loc != -1 and row != loc:
                continue
            z = z_ori.clone()
            for val in interpolation:
                z[:, row] = val
                sample = F.sigmoid(decoder(z)).data
                samples.append(sample)
                gifs.append(sample)

    samples = torch.cat(samples, dim=0).cpu()
    title = '{}_latent_traversal(iter:{})'.format(key, 1)

    output_dir = os.path.join(output_dir, str(1))
    mkdirs(output_dir)
    gifs = torch.cat(gifs)
    print("gif size is {}".format(gifs.shape))
    gifs = gifs.view(len(Z), z_dim, len(interpolation), 1, 256, 256).transpose(1, 2)

    for i, key in enumerate(Z.keys()):
        for j, val in enumerate(interpolation):
            save_image(tensor=gifs[i][j].cpu(),
                       filename=os.path.join(output_dir, '{}_{}.jpg'.format(key, j)),
                       nrow=z_dim, pad_value=1)

        # grid2gif(str(os.path.join(output_dir, key + '*.jpg')),
        #          str(os.path.join(output_dir, key + '.gif')), delay=10)




if __name__ == "__main__":

    ## Load the model
    # VAE = FactorVAE1(z_dim=20)
    # checkpoint = torch.load("checkpoints/run_scoor/370000")
    # print(checkpoint['model_states'].keys())
    # VAE.load_state_dict(checkpoint['model_states']["VAE"])
    # VAE.eval()
    #
    # dataset = return_cts("~/Desktop/SCOORTest/SCOOR")


    # namelist, Z = inference(VAE, dataset)
    # listarray = []
    # for name, z in zip(namelist, Z):
    #     line = [name]
    #     listarray.append(line + z.detach().numpy().tolist())
    #
    # # print(listarray)
    # df = pd.DataFrame(listarray)
    # print(df)
    # df.to_csv("FactorRep.csv", header=False, index=False)

    # reconstruct(VAE, dataset)


    ### Test the VAE256
    # VAE = FactorVAE256()
    # checkpoint = torch.load("checkpoints/runsccor256/runscoor256_jan27/10000")
    # print(checkpoint['model_states'].keys())
    # VAE.load_state_dict(checkpoint['model_states']["VAE"])
    # VAE.eval()
    #
    # dataset = return_cts("~/Desktop/SCOORTest/SCOOR", 256)
    # # reconstruct(VAE, dataset, 256)
    #
    #
    # namelist, Z = inference(VAE, dataset)
    # listarray = []
    # print("Z shape is {}".format(np.array(Z).shape))
    # for name, z in zip(namelist, Z):
    #     line = [name]
    #     listarray.append(line + z.detach().numpy().tolist())
    #
    # # print(listarray)
    # df = pd.DataFrame(listarray)
    # print(df)
    # df.to_csv("FactorRep256_64.csv", header=False, index=False)

    # reconstruct(VAE, dataset, 256)

    ## test on travers_Z
    VAE = FactorVAE256(z_dim=64)
    checkpoint = torch.load("checkpoints/runsccor256/10000")
    print(checkpoint['model_states'].keys())
    VAE.load_state_dict(checkpoint['model_states']["VAE"])
    VAE.eval()

    dataset = getdata("Visualization", 256)
    visualize_traverse(VAE, dataset)