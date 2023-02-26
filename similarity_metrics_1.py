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

from scipy.spatial.distance import cdist


def majority_vote_segmentation(segments):
    """
    Compute the majority vote segmentation from a list of binary segmentations.
    """
    num_segments, depth, height, width = segments.shape
    # Compute the sum of votes along the first axis
    votes = np.sum(segments, axis=0)
    # Compute the threshold for majority vote
    threshold = num_segments // 2 + 1
    # Apply the threshold to obtain the output segmentation
    result = votes >= threshold
    return result



def directed_hausdorff_distance(A, B):
    """
    Compute the directed Hausdorff distance from set A to set B
    """
    D = cdist(A, B)
    return np.max(np.min(D, axis=1))

def hausdorff_distance(A, B):
    """
    Compute the Hausdorff distance between set A and set B
    """
    return max(directed_hausdorff_distance(A, B), directed_hausdorff_distance(B, A))






def msd(gt_vertices, seg_vertices):
    # Compute distances from GT vertices to segmentation surface
    gt_to_seg = cdist(gt_vertices, seg_vertices)
    closest_seg_distances = np.min(gt_to_seg, axis=1)

    # Compute distances from segmentation vertices to GT surface
    seg_to_gt = cdist(seg_vertices, gt_vertices)
    closest_gt_distances = np.min(seg_to_gt, axis=1)


    # Compute MSD
    numerator = (np.sum(closest_gt_distances) / len(closest_gt_distances)) + (np.sum(closest_seg_distances) / len(closest_seg_distances))
    msd = numerator / 2

    return msd

# dice similarity function
def myDice(A, B):
    out = (2 * np.sum(A * B)) / (np.sum(A) + np.sum(B))
    return out

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
    surfs_gt = s.connectedComponents()
    gt_vol = s.volume(np.size(surfs_gt), surfs_gt)
    s.display(True)

    s = surface()
    s.color = cols(1 % 256)[0:3]
    s.opacity = 0.5
    s.createSurfaceFromVolume(imgzp_t1_msk, voxsz_t1_msk, 0.5)
    surfs_t1 = s.connectedComponents()
    t1_vol = s.volume(np.size(surfs_t1), surfs_t1)
    s.display(True)

    s = surface()
    s.color = (1, 1 ,1)
    s.opacity = 0.9
    s.createSurfaceFromVolume(imgzp_t2_msk, voxsz_t2_msk, 0.5)
    surfs_t2 = s.connectedComponents()
    t2_vol = s.volume(np.size(surfs_t2), surfs_t2)
    s.display(True)

    s = surface()
    s.color = (0,0,0)
    s.opacity = 0.1
    s.createSurfaceFromVolume(imgzp_t3_msk, voxsz_t3_msk, 0.5)
    surfs_t3 = s.connectedComponents()
    t3_vol = s.volume(np.size(surfs_t3), surfs_t3)
    s.display(True)

    # mlab.show()
    # # end of displaying the masks
    binary_image_gt = np.zeros_like(img_gt_msk)
    binary_image_gt[img_gt_msk > 0.5] = 1

    binary_image_t1 = np.zeros_like(img_t1_msk)
    binary_image_t1[img_t1_msk > 0.5] = 1

    binary_image_t2 = np.zeros_like(img_t2_msk)
    binary_image_t2[img_t2_msk > 0.5] = 1

    binary_image_t3 = np.zeros_like(img_t3_msk)
    binary_image_t3[img_t3_msk > 0.5] = 1

    # dice similarity coefficient for t1 image and the ground truth
    dice_sim_gt_t1 = myDice(binary_image_gt, binary_image_t1)
    # dice similarity coefficient for t1 image and the ground truth
    dice_sim_gt_t2 = myDice(binary_image_gt, binary_image_t2)
    # dice similarity coefficient for t1 image and the ground truth
    dice_sim_gt_t3 = myDice(binary_image_gt, binary_image_t3)

    print("Dice Similarity gt and t1 =", dice_sim_gt_t1)
    print("Dice Similarity gt and t2 =", dice_sim_gt_t2)
    print("Dice Similarity gt and t3 =", dice_sim_gt_t3)

    # mean symmetric absolute surface distance
    # gt and t1
    # mean from gt to t1
    msd_gt_t1 = msd(surfs_gt[0].verts, surfs_t1[0].verts)
    msd_gt_t2 = msd(surfs_gt[0].verts, surfs_t2[0].verts)
    msd_gt_t3 = msd(surfs_gt[0].verts, surfs_t3[0].verts)

    print("Mean symmetric absolute surface dist gt and t1 =", msd_gt_t1)
    print("Mean symmetric absolute surface dist gt and t2 =", msd_gt_t2)
    print("Mean symmetric absolute surface dist gt and t3 =", msd_gt_t3)

    hausdorff_gt_t1 = hausdorff_distance(surfs_gt[0].verts, surfs_t1[0].verts)
    hausdorff_gt_t2 = hausdorff_distance(surfs_gt[0].verts, surfs_t2[0].verts)
    hausdorff_gt_t3 = hausdorff_distance(surfs_gt[0].verts, surfs_t3[0].verts)

    print("Hausdorff dist gt and t1 =", hausdorff_gt_t1)
    print("Hausdorff dist gt and t2 =", hausdorff_gt_t2)
    print("Hausdorff dist gt and t3 =", hausdorff_gt_t3)


    maj_vot_seg = np.array([binary_image_t1, binary_image_t2, binary_image_t3])
    result = majority_vote_segmentation(maj_vot_seg)

    print("Majority vote segmentation:\n", result)

while (1):
    continue

