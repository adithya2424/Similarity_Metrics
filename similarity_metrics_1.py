import nrrd
import matplotlib.pyplot as plt
import numpy as np
from displayvol import *
from surface_adi_new import *
from mayavi import mlab
import pygame
import matplotlib as mp
from tqdm import tqdm
import pandas as pd
import matplotlib as mp

folders = ['0522c0001']
           # '0522c0002',
           # '0522c0003', '0522c0009', '0522c0013',
           # '0522c0014', '0522c0017', '0522c0057',
           # '0522c0070', '0522c0081']

for folder_name in tqdm(folders):
    # load a ground truth msk
    gt_msk = '/home-local/adi/scripts/EECE_395/' + \
                folder_name + '/structures/Mandible.nrrd'
    img_gt_msk, imgh = nrrd.read(gt_msk)
    voxsz_gt_msk = [imgh['space directions'][0][0], imgh['space directions'][1][1], imgh['space directions'][2][2]]
    imgzp = np.zeros((np.array(np.shape(img_gt_msk)) + 2))
    imgzp[1:-1, 1:-1, 1:-1] = img_gt_msk
    imgzp_gt_msk = imgzp

    # load a target 1 msk
    t1_msk = '/home-local/adi/scripts/EECE_395/' + \
             folder_name + '/structures/target1.nrrd'
    img_t1_msk, imgh = nrrd.read(t1_msk)
    voxsz_t1_msk = [imgh['space directions'][0][0], imgh['space directions'][1][1], imgh['space directions'][2][2]]
    imgzp = np.zeros((np.array(np.shape(img_t1_msk)) + 2))
    imgzp[1:-1, 1:-1, 1:-1] = img_t1_msk
    imgzp_t1_msk = imgzp

    # load a target 2 msk
    t2_msk = '/home-local/adi/scripts/EECE_395/' + \
             folder_name + '/structures/target2.nrrd'
    img_t2_msk, imgh = nrrd.read(t2_msk)
    voxsz_t2_msk = [imgh['space directions'][0][0], imgh['space directions'][1][1], imgh['space directions'][2][2]]
    imgzp = np.zeros((np.array(np.shape(img_t2_msk)) + 2))
    imgzp[1:-1, 1:-1, 1:-1] = img_t2_msk
    imgzp_t2_msk = imgzp

    # load a target 3 msk
    t3_msk = '/home-local/adi/scripts/EECE_395/' + \
             folder_name + '/structures/target3.nrrd'
    img_t3_msk, imgh = nrrd.read(t3_msk)
    voxsz_t3_msk = [imgh['space directions'][0][0], imgh['space directions'][1][1], imgh['space directions'][2][2]]
    imgzp = np.zeros((np.array(np.shape(img_t3_msk)) + 2))
    imgzp[1:-1, 1:-1, 1:-1] = img_t3_msk
    imgzp_t3_msk = imgzp


    cols = mp.colormaps['jet']
    mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
    # display four masks
    s = surface()
    s.color = cols(80 % 256)[0:3]
    s.opacity = 0.5
    s.createSurfaceFromVolume(imgzp_gt_msk, voxsz_gt_msk, 0.5)
    surfs = s.connectedComponents()
    gt_vol = s.volume(np.size(surfs), surfs)
    s.display(True)

    s = surface()
    s.color = cols(1 % 256)[0:3]
    s.opacity = 0.5
    s.createSurfaceFromVolume(imgzp_t1_msk, voxsz_t1_msk, 0.5)
    surfs = s.connectedComponents()
    t1_vol = s.volume(np.size(surfs), surfs)
    s.display(True)

    s = surface()
    s.color = (1, 1 ,1)
    s.opacity = 0.9
    s.createSurfaceFromVolume(imgzp_t2_msk, voxsz_t2_msk, 0.5)
    surfs = s.connectedComponents()
    t2_vol = s.volume(np.size(surfs), surfs)
    s.display(True)

    s = surface()
    s.color = (0,0,0)
    s.opacity = 0.1
    s.createSurfaceFromVolume(imgzp_t3_msk, voxsz_t3_msk, 0.5)
    surfs = s.connectedComponents()
    t3_vol = s.volume(np.size(surfs), surfs)
    s.display(True)

    # mlab.show()
    # # end of displaying the masks

    # dice similarity between gt and each of three raters
    intersection = np.array(gt_vol) * np.array(t1_vol)
    union = np.array(gt_vol) + np.array(t1_vol)
    dice_coef = 2 * intersection / union
    print('Dice similarity coefficient between GT and T1:', dice_coef)


while (1):
    continue

