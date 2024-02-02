import os

# from PIL import Image
# import torchvision.transforms as transforms
# import torch
# a=Image.open('D:\pytorch_\PyTorch_tudui\hymenoptera_data\\train\\ants\\0013035.jpg')
# # print(a.shape)
# c=a.mode
# print(c)
# c=a.format
# print(c)
# d=a.size
# print(d)
# e=a.getbands()
# print(e)
# b=transforms.ToTensor()
# a=b(a)
# print(a.shape)
import numpy as np
#
# # 创建两个1D NumPy数组
# x = np.array([1, 2, 3])
# y = np.array([4, 5, 6])
# print(x.shape,y.shape)
# # 执行广播操作，将它们扩展为2D数组
# x = x[None, ...]
# y = y[None, ...]
# print(x.shape,y.shape)
# import nibabel as nib
# from torchvision.transforms import transforms
# import medpy.io as medio
# flair, flair_header = medio.load('D:\download\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_flair.nii')
# t1ce, t1ce_header = medio.load('D:\download\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_t1ce.nii')
# t1, t1_header = medio.load('D:\download\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_t1.nii')
# t2, t2_header = medio.load('D:\download\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_t2.nii')
# print(flair.shape)
# print(t1ce.shape)
# print(t1.shape)
# print(t2.shape)
# vol1 = np.stack((flair, t1ce, t1, t2), axis=0).astype(np.float32)
# print(vol1.shape)
# vol1 = vol1.transpose(1,2,3,0)
# print (vol1.shape)
# x=nib.load('D:\download\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_flair.nii')
# x=x.get_fdata()
# print(x.shape)
# import SimpleITK as sitk
# a=sitk.ReadImage('D:\download\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_flair.nii')
# print(a.GetWidth(), a.GetHeight(), a.GetDepth())
# a=sitk.GetArrayFromImage(a)
#
# print(a.shape)
# import nibabel as nib
# a=nib.load('D:\download\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_flair.nii')
# a=a.get_fdata()
# print(a.shape)

# import nibabel as nib
#
# # 指定NIfTI文件路径
# nii_file_path = 'D:\download\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_flair.nii'
#
# # 读取NIfTI文件
# img = nib.load(nii_file_path)
#
# # 获取头文件中的维度信息
# header = img.header
# dim = header.get_data_shape()
#
# # 打印维度信息
# print("图像维度信息：")
# print(f"高度（dim[0]）: {dim[0]}")
# print(f"宽度（dim[1]）: {dim[1]}")
# print(f"深度（dim[2]）: {dim[2]}")
#------------------------------------------
# import os
# import numpy as np
# import medpy.io as medio
#
# # src_path = 'D:\download\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData'    #'path_of_raw_BRATS2020'
# # tar_path = 'D:\download\BraTS2020_TrainingData'            #'path_of_processed_BRATS2020'
#
# # name_list = os.listdir(src_path)
#
# def sup_128(xmin, xmax):
#     if xmax - xmin < 128:
#         print ('#' * 100)
#         ecart = int((128-(xmax-xmin))/2)
#         xmax = xmax+ecart+1
#         xmin = xmin-ecart
#     if xmin < 0:
#         xmax-=xmin
#         xmin=0
#     return xmin, xmax
#
# def crop(vol):
#     if len(vol.shape) == 4:
#         vol = np.amax(vol, axis=0)
#     assert len(vol.shape) == 3
#
#     x_dim, y_dim, z_dim = tuple(vol.shape)
#     x_nonzeros, y_nonzeros, z_nonzeros = np.where(vol != 0)
#
#     x_min, x_max = np.amin(x_nonzeros), np.amax(x_nonzeros)
#     y_min, y_max = np.amin(y_nonzeros), np.amax(y_nonzeros)
#     z_min, z_max = np.amin(z_nonzeros), np.amax(z_nonzeros)
#
#     x_min, x_max = sup_128(x_min, x_max)
#     y_min, y_max = sup_128(y_min, y_max)
#     z_min, z_max = sup_128(z_min, z_max)
#
#     return x_min, x_max, y_min, y_max, z_min, z_max
#
# def normalize(vol):
#     mask = vol.sum(0) > 0
#     for k in range(4):
#         x = vol[k, ...]
#         y = x[mask]
#         x = (x - y.mean()) / y.std()
#         vol[k, ...] = x
#
#     return vol

# if not os.path.exists(os.path.join(tar_path, 'vol')):
#     os.makedirs(os.path.join(tar_path, 'vol'))

# if not os.path.exists(os.path.join(tar_path, 'seg')):
#     os.makedirs(os.path.join(tar_path, 'seg'))

# for file_name in name_list:
# print (file_name)
# num = file_name.split('_')[2]
# HLG = 'HG_' if int(num) <= 259 or int(num) >= 336 else 'LG_'
# flair, flair_header = medio.load('D:\download\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_flair.nii')
# t1ce, t1ce_header = medio.load('D:\download\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_t1ce.nii')
# t1, t1_header = medio.load('D:\download\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_t1.nii')
# t2, t2_header = medio.load('D:\download\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_t2.nii')
#
# vol = np.stack((flair, t1ce, t1, t2), axis=0).astype(np.float32)#将各个模态堆叠起来
# x_min, x_max, y_min, y_max, z_min, z_max = crop(vol)
# vol1 = normalize(vol[:, x_min:x_max, y_min:y_max, z_min:z_max])
# vol1 = vol1.transpose(1,2,3,0)
# print(vol1.shape)
#
# seg, seg_header = medio.load('D:\download\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_seg.nii')
# seg = seg.astype(np.uint8)
# seg1 = seg[x_min:x_max, y_min:y_max, z_min:z_max]
#
# print(seg1.shape)
#np.save(os.path.join(tar_path, 'vol', HLG+file_name+'_vol.npy'), vol1)
#p.save(os.path.join(tar_path, 'seg', HLG+file_name+'_seg.npy'), seg1)



"""
@author: Chenggang
@github: https://github.com/MissShihongHowRU
@time: 2020-09-09 22:04
"""
# from torch.utils.data import Dataset
# import numpy as np
# import os
# import numpy as np
#
# np.random.seed(0)
#
# import random
#
# random.seed(0)
#
#
# # from random import sample
#
#
# def validation_sampling(data_list, test_size=0.2):
#     n = len(data_list)
#     m = int(n * test_size)
#     val_items = random.sample(data_list, m)
#     tr_items = list(set(data_list) - set(val_items))
#     return tr_items, val_items
#
#
# def random_intensity_shift(imgs_array, brain_mask, limit=0.1):
#     """
#     Only do intensity shift on brain voxels
#     :param imgs_array: The whole input image with shape of (4, 155, 240, 240)
#     :param brain_mask:
#     :param limit:
#     :return:
#     """
#
#     shift_range = 2 * limit
#     for i in range(len(imgs_array) - 1):
#         factor = -limit + shift_range * np.random.random()
#         std = imgs_array[i][brain_mask].std()
#         imgs_array[i][brain_mask] = imgs_array[i][brain_mask] + factor * std
#     return imgs_array
#
#
# def random_scale(imgs_array, brain_mask, scale_limits=(0.9, 1.1)):
#     """
#     Only do random_scale on brain voxels
#     :param imgs_array: The whole input image with shape of (4, 155, 240, 240)
#     :param scale_limits:
#     :return:
#     """
#     scale_range = scale_limits[1] - scale_limits[0]
#     for i in range(len(imgs_array) - 1):
#         factor = scale_limits[0] + scale_range * np.random.random()
#         imgs_array[i][brain_mask] = imgs_array[i][brain_mask] * factor
#     return imgs_array
#
#
# def random_mirror_flip(imgs_array, prob=0.5):
#     """
#     Perform flip along each axis with the given probability; Do it for all voxels；
#     labels should also be flipped along the same axis.
#     :param imgs_array:
#     :param prob:
#     :return:
#     """
#     for axis in range(1, len(imgs_array.shape)):
#         random_num = np.random.random()
#         if random_num >= prob:
#             if axis == 1:
#                 imgs_array = imgs_array[:, ::-1, :, :]
#             if axis == 2:
#                 imgs_array = imgs_array[:, :, ::-1, :]
#             if axis == 3:
#                 imgs_array = imgs_array[:, :, :, ::-1]
#     return imgs_array
#
#
# def random_crop(imgs_array, crop_size=(128, 192, 160), lower_limit=(0, 32, 40)):
#     """
#     crop the image ((155, 240, 240) for brats data) into the crop_size
#     the random area is now limited at (0:155, 32:224, 40:200), by default
#     :param imgs_array:
#     :param crop_size:
#     :return:
#     """
#     orig_shape = np.array(imgs_array.shape[1:])
#     crop_shape = np.array(crop_size)
#     # ranges = np.array(orig_shape - crop_shape, dtype=np.uint8)
#     # lower_limits = np.random.randint(np.array(ranges))
#     lower_limit_z = np.random.randint(lower_limit[0], 155 - crop_size[0])
#     if crop_size[1] < 192:
#         lower_limit_y = np.random.randint(lower_limit[1], 224 - crop_size[1])
#     else:
#         lower_limit_y = np.random.randint(0, 240 - crop_size[1])
#     if crop_size[2] < 160:
#         lower_limit_x = np.random.randint(lower_limit[2], 200 - crop_size[2])
#     else:
#         lower_limit_x = np.random.randint(0, 240 - crop_size[2])
#     lower_limits = np.array((lower_limit_z, lower_limit_y, lower_limit_x))
#     upper_limits = lower_limits + crop_shape
#     imgs_array = imgs_array[:, lower_limits[0]: upper_limits[0],
#                  lower_limits[1]: upper_limits[1], lower_limits[2]: upper_limits[2]]
#     return imgs_array
#
#
# def validation_time_crop(imgs_array, crop_size=(128, 192, 160)):
#     """
#     crop the image ((155, 240, 240) for brats data) into the crop_size
#     :param imgs_array:
#     :param crop_size:
#     :return:
#     """
#     orig_shape = np.array(imgs_array.shape[1:])
#     crop_shape = np.array(crop_size)
#     lower_limit_z = np.random.randint(orig_shape[0] - crop_size[0])
#     center_y = 128
#     center_x = 120
#     lower_limit_y = center_y - crop_size[-2] // 2  # (128, 160, 128)  (?, 48, 56)
#     lower_limit_x = center_x - crop_size[-1] // 2  # (128, 192, 160)  (?, 32, 40)
#     lower_limits = np.array((lower_limit_z, lower_limit_y, lower_limit_x))
#
#     upper_limits = lower_limits + crop_shape
#
#     imgs_array = imgs_array[:, lower_limits[0]: upper_limits[0],
#                  lower_limits[1]: upper_limits[1], lower_limits[2]: upper_limits[2]]
#     return imgs_array
#
#
# def test_time_crop(imgs_array, crop_size=(144, 192, 160)):
#     """
#     crop the test image around the center; default crop_zise change from (128, 192, 160) to (144, 192, 160)
#     :param imgs_array:
#     :param crop_size:
#     :return: image with the size of crop_size
#     """
#     orig_shape = np.array(imgs_array.shape[1:])
#     crop_shape = np.array(crop_size)
#     center = orig_shape // 2
#     lower_limits = center - crop_shape // 2  # (13, 24, 40) (5, 24, 40)
#     upper_limits = center + crop_shape // 2  # (141, 216, 200) (149, 216, 200）
#     # upper_limits = lower_limits + crop_shape
#     imgs_array = imgs_array[:, lower_limits[0]: upper_limits[0],
#                  lower_limits[1]: upper_limits[1], lower_limits[2]: upper_limits[2]]
#     return imgs_array
#
#
# def test_time_flip(imgs_array, tta_idx):
#     if tta_idx == 0:  # [0, 0, 0]
#         return imgs_array
#     if tta_idx == 1:  # [1, 0, 0]
#         return imgs_array[:, ::-1, :, :]
#     if tta_idx == 2:  # [0, 1, 0]
#         return imgs_array[:, :, ::-1, :]
#     if tta_idx == 3:  # [0, 0, 1]
#         return imgs_array[:, :, :, ::-1]
#     if tta_idx == 4:  # [1, 1, 0]
#         return imgs_array[:, ::-1, ::-1, :]
#     if tta_idx == 5:  # [1, 0, 1]
#         return imgs_array[:, ::-1, :, ::-1]
#     if tta_idx == 6:  # [0, 1, 1]
#         return imgs_array[:, :, ::-1, ::-1]
#     if tta_idx == 7:  # [1, 1, 1]
#         return imgs_array[:, ::-1, ::-1, ::-1]
#
#
# def preprocess_label(img, single_label=None):
#     """
#     Separates out the 3 labels from the segmentation provided, namely:
#     GD-enhancing tumor (ET — label 4), the peritumoral edema (ED — label 2))
#     and the necrotic and non-enhancing tumor core (NCR/NET — label 1)
#     """
#
#     ncr = img == 1  # Necrotic and Non-Enhancing Tumor (NCR/NET) - orange
#     ed = img == 2  # Peritumoral Edema (ED) - yellow
#     et = img == 4  # GD-enhancing Tumor (ET) - blue
#     if not single_label:
#         # return np.array([ncr, ed, et], dtype=np.uint8)
#         return np.array([ed, ncr, et], dtype=np.uint8)
#     elif single_label == "WT":
#         img[ed] = 1
#         img[et] = 1
#     elif single_label == "TC":
#         img[ncr] = 0
#         img[ed] = 1
#         img[et] = 1
#     elif single_label == "ET":
#         img[ncr] = 0
#         img[ed] = 0
#         img[et] = 1
#     else:
#         raise RuntimeError("the 'single_label' type must be one of WT, TC, ET, and None")
#     return img[np.newaxis, :]
#
#
# class BratsDataset(Dataset):
#     def __init__(self, phase, config):
#         super(BratsDataset, self).__init__()
#
#         self.config = config
#         self.phase = phase
#         self.input_shape = config["input_shape"]
#         self.data_path = config["data_path"]
#         self.seg_label = config["seg_label"]
#         self.intensity_shift = config["intensity_shift"]
#         self.scale = config["scale"]
#         self.flip = config["flip"]
#
#         if phase == "train":
#             self.patient_names = config["training_patients"]  # [:4]
#         elif phase == "validate" or phase == "evaluation":
#             self.patient_names = config["validation_patients"]  # [:2]
#         elif phase == "test":
#             self.test_path = config["test_path"]
#             self.patient_names = config["test_patients"]
#             self.tta_idx = config["tta_idx"]
#
#     def __getitem__(self, index):
#         patient = self.patient_names[index]
#         self.file_path = os.path.join(self.data_path, 'npy', patient + ".npy")
#         if self.phase == "test":
#             self.file_path = os.path.join(self.test_path, 'npy', patient + ".npy")
#         imgs_npy = np.load(self.file_path)
#
#         if self.phase == "train":
#             nonzero_masks = [i != 0 for i in imgs_npy[:-1]]
#             brain_mask = np.zeros(imgs_npy.shape[1:], dtype=bool)
#             for chl in range(len(nonzero_masks)):
#                 brain_mask = brain_mask | nonzero_masks[chl]  # (155, 240, 240)
#             # data augmentation
#             cur_image_with_label = imgs_npy.copy()
#             cur_image = cur_image_with_label[:-1]
#             if self.intensity_shift:
#                 cur_image = random_intensity_shift(cur_image, brain_mask)
#             if self.scale:
#                 cur_image = random_scale(cur_image, brain_mask)
#
#             cur_image_with_label[:-1] = cur_image
#             cur_image_with_label = random_crop(cur_image_with_label, crop_size=self.input_shape[2:])
#
#             if self.flip:  # flip should be performed with labels
#                 cur_image_with_label = random_mirror_flip(cur_image_with_label)
#
#         elif self.phase == "validate":
#             # cur_image_with_label = validation_time_crop(imgs_npy)
#             cur_image_with_label = validation_time_crop(imgs_npy, crop_size=self.input_shape[2:])
#
#         elif self.phase == "evaluation":
#             cur_image_with_label = imgs_npy.copy()
#
#         if self.phase == "validate" or self.phase == "train" or self.phase == "evaluation":
#             inp_data = cur_image_with_label[:-1]
#             seg_label = preprocess_label(cur_image_with_label[-1], self.seg_label)
#             if self.config["VAE_enable"]:
#                 final_label = np.concatenate((seg_label, inp_data), axis=0)
#             else:
#                 final_label = seg_label
#
#             return np.array(inp_data), np.array(final_label)
#
#         elif self.phase == "test":
#             imgs_npy = test_time_crop(imgs_npy)
#             if self.config["predict_from_train_data"]:
#                 imgs_npy = imgs_npy[:-1]
#             imgs_npy = test_time_flip(imgs_npy, self.tta_idx)
#             # np.save("../test_time_crop/{}.npy".format(str(index)), imgs_npy)
#             # only use when doing inference for training-data
#             # imgs_npy = imgs_npy[:4, :, :, :]
#             return np.array(imgs_npy)
#
#     # np.array() solve the problem of "ValueError: some of the strides of a given numpy array are negative"
#
#     def __len__(self):
#         return len(self.patient_names)




# import nibabel as nib
# import numpy as np
#
# # 准备像素数据
# data = np.load('D:\\brats\BRATS2020_Training_none_npy\\vol\HG_BraTS20_Training_001_vol.npy')  # 这里使用随机数据替代真实的图像数据
# print(data.shape)
# # 设置空间信息
# affine = np.eye(4)  # 单位仿射矩阵，表示图像的初始定位和方向
#
# # 创建NIfTI图像对象
# nifti_img = nib.Nifti1Image(data, affine)
#
# # 保存图像到文件
# nib.save(nifti_img, 'D:\\brats\BRATS2020_Training_none_npy\\1')

