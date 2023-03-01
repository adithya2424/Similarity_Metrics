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
from sklearn.metrics import confusion_matrix
import seaborn as sb
import scipy as sp

folders = ['0522c0001']
           # '0522c0002',
           # '0522c0003', '0522c0009', '0522c0013',
           # '0522c0014', '0522c0017', '0522c0057',
           # '0522c0070', '0522c0081']

def foreground_sensitivity_specificity(mask_true, mask_pred):
    true_pos = np.sum(np.logical_and(mask_true == 1, mask_pred == 1))
    true_neg = np.sum(np.logical_and(mask_true == 0, mask_pred == 0))
    false_pos = np.sum(np.logical_and(mask_true == 0, mask_pred == 1))
    false_neg = np.sum(np.logical_and(mask_true == 1, mask_pred == 0))

    # Compute the foreground sensitivity and specificity
    foreground_sensitivity = true_pos / (true_pos + false_neg)
    foreground_specificity = true_neg / (true_neg + false_pos)

    return foreground_sensitivity, foreground_specificity

def confusion_matrix(mask_true, mask_pred):
    true_pos = np.sum(np.logical_and(mask_true == 1, mask_pred == 1))
    true_neg = np.sum(np.logical_and(mask_true == 0, mask_pred == 0))
    false_pos = np.sum(np.logical_and(mask_true == 0, mask_pred == 1))
    false_neg = np.sum(np.logical_and(mask_true == 1, mask_pred == 0))

    conf_mat = np.array([[true_pos, false_pos], [false_neg, true_neg]])

    return conf_mat

# dice similarity function
def myDice(A, B):
    out = (2 * np.sum(A * B)) / (np.sum(A) + np.sum(B))
    return out

def surafce_distance_clone(surfaces_base, surfaces_reference):
    # Convert faces to int type
    surfaces_base.faces = np.array(surfaces_base.faces, dtype='int')
    surfaces_reference.faces = np.array(surfaces_reference.faces, dtype='int')
    # Base to reference distance
    min_value_pt = np.min(surfaces_base.faces)
    max_value_pt = np.max(surfaces_base.faces)
    p = surfaces_base.verts[:, :]
    q = surfaces_reference.verts[:, :]
    faces = surfaces_reference.faces
    q1 = q[faces[:, 0], :]
    q2 = q[faces[:, 1], :]
    q3 = q[faces[:, 2], :]
    v1 = q2 - q1
    v2 = q3 - q1
    v3 = q3 - q2
    V = np.concatenate((v1[:, :, np.newaxis], v2[:, :, np.newaxis]), axis=2)
    pinv = np.linalg.pinv(V)
    mind = []
    for psub in p:
        temp = 100000
        buff = (psub - q1)[:, :, np.newaxis]
        ab = pinv @ buff
        ab_sub_mask = (ab[:, 0,:] >= 0) & (ab[:, 1,:] >= 0) & (np.sum(ab, axis=1) <= 1)
        w = V @ ab + q1[:, :, np.newaxis]
        w_diff = w - psub[:, np.newaxis]
        pt_dist = np.linalg.norm(w_diff, axis=1)
        pt_dist = ab_sub_mask * pt_dist
        c = np.sum(v1[:, :] * (psub - q1[:, :]), axis=1) / np.sum(v1[:, :] * v1[:, :], axis=1)
        d = np.sum(v2[:, :] * (psub - q1[:, :]), axis=1) / np.sum(v2[:, :] * v2[:, :], axis=1)
        e = np.sum(v3[:, :] * (psub - q2[:, :]), axis=1) / np.sum(v3[:, :] * v3[:, :], axis=1)
        c = ~ab_sub_mask * c[:, np.newaxis]
        d = ~ab_sub_mask * d[:, np.newaxis]
        e = ~ab_sub_mask * e[:, np.newaxis]
        c = np.clip(c, 0, 1)
        d = np.clip(d, 0, 1)
        e = np.clip(e, 0, 1)
        d1 = np.sqrt(np.sum((q1[:, :] + c[:, :] * v1[:, :] - psub) ** 2, axis=1))
        d2 = np.sqrt(np.sum((q1[:, :] + d[:, :] * v2[:, :] - psub) ** 2, axis=1))
        d3 = np.sqrt(np.sum((q2[:, :] + e[:, :] * v3[:, :] - psub) ** 2, axis=1))
        pt_dist_2 = np.min([d1, d2, d3], axis=0)
        pt_dist_2 = pt_dist_2[:, np.newaxis]
        pt_dist_final = np.min(pt_dist + pt_dist_2)
        mind.append(pt_dist_final)
    return np.mean(np.array(mind))


#
#
#
#     # ab = np.matmul(pinv, buff)
#     # is_valid = (ab[0] >= 0) & (ab[1] >= 0) & (np.sum(ab, axis=0) <= 1)
#     # w = np.matmul(V[:, :, np.newaxis], ab[:, is_valid, np.newaxis]) + q1.T[np.newaxis, :, np.newaxis]
#     # pt_dist = np.linalg.norm(w - p[:, :, np.newaxis], axis=1)
#     #
#     # c = np.sum(v1[is_valid] * (p[is_valid] - q1[faces[is_valid, 0]]), axis=1) / np.sum(v1[is_valid] ** 2, axis=1)
#     # d = np.sum(v2[is_valid] * (p[is_valid] - q1[faces[is_valid, 0]]), axis=1) / np.sum(v2[is_valid] ** 2, axis=1)
#     # e = np.sum((q3 - q2)[is_valid] * (p[is_valid] - q2[faces[is_valid, 1]]), axis=1) / np.sum((q3 - q2)[is_valid] ** 2,
#     #                                                                                           axis=1)
#     #
#     # c[c < 0] = 0
#     # c[c > 1] = 1
#     # d[d < 0] = 0
#     # d[d > 1] = 1
#     # e[e < 0] = 0
#     # e[e > 1] = 1
#     #
#     # d1 = np.sqrt(np.sum((q1[faces[is_valid, 0]] + c[:, np.newaxis] * v1[is_valid] - p[is_valid]) ** 2, axis=1))
#     # d2 = np.sqrt(np.sum((q1[faces[is_valid, 0]] + d[:, np.newaxis] * v2[is_valid] - p[is_valid]) ** 2, axis=1))
#     # d3 = np.sqrt(np.sum((q2[faces[is_valid, 1]] + e[:, np.newaxis] * (q3 - q2)[is_valid] - p[is_valid]) ** 2, axis=1))
#     # pt_dist[~is_valid] = np.min([d1, d2, d3], axis=0)[~is_valid]
#
#     dist_vector_base_reference = np.mean(pt_dist, axis=1)
#
#
# def surface_distance(surfaces_base, surfaces_reference):
#     surfaces_base[0].faces = np.array(surfaces_base[0].faces, dtype='int')
#     surfaces_reference[0].faces = np.array(surfaces_base[0].faces, dtype='int')
#     # base to reference distance
#     min_value_pt = np.min(surfaces_base[0].faces)
#     max_value_pt = np.max(surfaces_base[0].faces)
#     p = surfaces_base[min_value_pt].verts[0, :]
#     dist_vector_base_reference = []
#     for p_index in range(0, max_value_pt):
#         p = surfaces_base[min_value_pt].verts[p_index, :]
#         # below loop computes distance vector of one point distance vector base to reference
#         temp = 1000000
#         for face in surfaces_reference[0].faces:
#             q1 = surfaces_reference[0].verts[face[0], :]
#             q2 = surfaces_reference[0].verts[face[1], :]
#             q3 = surfaces_reference[0].verts[face[2], :]
#             v1 = q2 - q1
#             v2 = q3 - q1
#             V = np.concatenate((v1[:, np.newaxis], v2[:, np.newaxis]), axis=1)
#             pinv = np.linalg.pinv(V)
#             buff = ((p - q1)[:, np.newaxis])
#             ab = pinv @ buff
#
#             if ab[0] >= 0 and ab[1] >= 0 and np.sum(ab) <= 1:
#                 w = V @ ab + q1[:, np.newaxis]
#                 pt_dist = np.linalg.norm(w - p[:, np.newaxis])
#
#             else:
#                 v3 = q3 - q2
#                 c = np.sum(v1 * (p - q1)) / np.sum(v1 * v1)
#                 d = np.sum(v2 * (p - q1)) / np.sum(v2 * v2)
#                 e = np.sum(v3 * (p - q2)) / np.sum(v3 * v3)
#                 if c < 0: c = 0
#                 if c > 1: c = 1
#                 if d < 0: d = 0
#                 if d > 1: d = 1
#                 if e < 0: e = 0
#                 if e > 1: e = 1
#                 d1 = np.sqrt(np.sum((q1 + c * v1 - p) ** 2))
#                 d2 = np.sqrt(np.sum((q1 + d * v2 - p) ** 2))
#                 d3 = np.sqrt(np.sum((q2 + e * v3 - p) ** 2))
#                 pt_dist = np.min([d1, d2, d3])
#
#             if pt_dist < temp:
#                 temp = pt_dist
#         dist_vector_base_reference.append(temp)
#         print(temp)
#     return np.mean(dist_vector_base_reference)



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
    t2_msk =  '/home-local/adi/scripts/EECE_395/' + \
                folder_name + '/structures/target2.nrrd'
    img_t2_msk, imgh = nrrd.read(t2_msk)
    voxsz_t2_msk = [imgh['space directions'][0][0], imgh['space directions'][1][1], imgh['space directions'][2][2]]
    imgzp = np.zeros((np.array(np.shape(img_t2_msk)) + 2))
    imgzp[1:-1, 1:-1, 1:-1] = img_t2_msk
    imgzp_t2_msk = imgzp

    # load a target 3 msk
    t3_msk =  '/home-local/adi/scripts/EECE_395/' + \
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
    gt_surfs = s
    surfs_gt = s.connectedComponents()
    gt_vol = np.array(s.volume(np.size(surfs_gt), surfs_gt))
    s.display(True)

    s = surface()
    s.color = cols(1 % 256)[0:3]
    s.opacity = 0.5
    s.createSurfaceFromVolume(imgzp_t1_msk, voxsz_t1_msk, 0.5)
    t1_surfs = s
    surfs_t1 = s.connectedComponents()
    t1_vol = np.array(s.volume(np.size(surfs_t1), surfs_t1))
    s.display(True)

    s = surface()
    s.color = (1, 1 ,1)
    s.opacity = 0.9
    s.createSurfaceFromVolume(imgzp_t2_msk, voxsz_t2_msk, 0.5)
    t2_surfs = s
    surfs_t2 = s.connectedComponents()
    t2_vol = np.array(s.volume(np.size(surfs_t2), surfs_t2))
    s.display(True)

    s = surface()
    s.color = (0,0,0)
    s.opacity = 0.1
    s.createSurfaceFromVolume(imgzp_t3_msk, voxsz_t3_msk, 0.5)
    t3_surfs = s
    surfs_t3 = s.connectedComponents()
    t3_vol = np.array(s.volume(np.size(surfs_t3), surfs_t3))
    s.display(True)

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

    # Compute the confusion matrix
    conf_mat_gt_t1 = confusion_matrix(imgzp_gt_msk, imgzp_t1_msk)
    conf_mat_gt_t2 = confusion_matrix(imgzp_gt_msk, imgzp_t2_msk)
    conf_mat_gt_t3 = confusion_matrix(imgzp_gt_msk, imgzp_t3_msk)

    # Print the confusion matrix
    print("Confusion matrix gt and t1:")
    print(conf_mat_gt_t1)
    print("Confusion matrix gt and t2:")
    print(conf_mat_gt_t2)
    print("Confusion matrix gt and t3:")
    print(conf_mat_gt_t3)

    # majority vote
    mv = img_t1_msk + img_t2_msk + img_t3_msk > 1.5
    s = surface()
    s.color = cols(50 % 256)[0:3]
    s.opacity = 0.5
    s.createSurfaceFromVolume(mv, voxsz_gt_msk, 0.5)
    mv_surfs = s
    surfs_mv = s.connectedComponents()
    mv_vol = np.array(s.volume(np.size(surfs_mv), surfs_mv))
    s.display(True)
    binary_image_mv = np.zeros_like(mv)
    binary_image_mv[mv > 0.5] = 1
    dice_sim_gt_mv = myDice(binary_image_gt, binary_image_mv)
    print("Dice Similarity gt and mv =", dice_sim_gt_mv)

    # mlab.show()
    sensitivity_gt_t1, specificity_gt_t1 = foreground_sensitivity_specificity(binary_image_gt, binary_image_t1)
    sensitivity_gt_t2, specificity_gt_t2 = foreground_sensitivity_specificity(binary_image_gt, binary_image_t2)
    sensitivity_gt_t3, specificity_gt_t3 = foreground_sensitivity_specificity(binary_image_gt, binary_image_t3)
    print("sensitity, specifictiy  gt and t1 =", sensitivity_gt_t1, specificity_gt_t1)
    print("sensitity, specifictiy  gt and t2 =", sensitivity_gt_t2, specificity_gt_t2)
    print("sensitity, specifictiy  gt and t3 =", sensitivity_gt_t3, specificity_gt_t3)


    # volumes = np.array([gt_vol, t1_vol, t2_vol, t3_vol])
    # sb.boxplot(volumes)
    # plt.show()
    # mean_gt_t1 = surafce_distance_clone(gt_surfs, t1_surfs)
    # mean_t1_gt = surafce_distance_clone(t1_surfs, gt_surfs)
    # masd_t1vgt = (mean_gt_t1 + mean_t1_gt)/2
    # HD_T1vgt = max(mean_gt_t1, mean_t1_gt)
    # print(f'MSASD gt vs t1: {masd_t1vgt}')
    # print(f'HD gt vs t1: {HD_T1vgt}')
    #
    # mean_gt_t2 = surafce_distance_clone(gt_surfs, t2_surfs)
    # mean_t2_gt = surafce_distance_clone(t2_surfs, gt_surfs)
    # masd_t2vgt = (mean_gt_t2 + mean_t2_gt) / 2
    # HD_T2vgt = max(mean_gt_t2, mean_t2_gt)
    # print(f'MSASD gt vs t2: {masd_t2vgt}')
    # print(f'HD gt vs t2: {HD_T2vgt}')
    #
    # mean_gt_t3 = surafce_distance_clone(gt_surfs, t3_surfs)
    # mean_t3_gt = surafce_distance_clone(t3_surfs, gt_surfs)
    # masd_t3vgt = (mean_gt_t3 + mean_t3_gt) / 2
    # HD_T3vgt = max(mean_gt_t3, mean_t3_gt)
    # print(f'MSASD gt vs t3: {masd_t3vgt}')
    # print(f'HD gt vs t3: {HD_T3vgt}')
    #
    # mean_gt_mv = surafce_distance_clone(gt_surfs, mv_surfs)
    # mean_mv_gt = surafce_distance_clone(mv_surfs, gt_surfs)
    # masd_mvvgt = (mean_gt_mv + mean_mv_gt) / 2
    # HD_mvvgt = max(mean_gt_mv, mean_mv_gt)
    # print(f'MSASD gt vs mv: {masd_mvvgt}')
    # print(f'HD gt vs mv: {HD_mvvgt}')

    mv_test = np.zeros((np.array(np.shape(mv)) + 2))
    mv_test[1:-1, 1:-1, 1:-1] = mv
    # wilcoxon gt, t1, t2, t3
    print(sp.stats.wilcoxon(imgzp_gt_msk.ravel(), imgzp_t1_msk.ravel()))
    print(sp.stats.wilcoxon(imgzp_gt_msk.ravel(), imgzp_t2_msk.ravel()))
    print(sp.stats.wilcoxon(imgzp_gt_msk.ravel(), imgzp_t3_msk.ravel()))
    print(sp.stats.wilcoxon(imgzp_gt_msk.ravel(), mv_test.ravel()))




while (1):
    continue
