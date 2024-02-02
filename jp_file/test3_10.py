# import nibabel as nib
# import numpy as np
#
# # 读取 NIfTI 文件
# nii_file_path = 'D:/dataset/brats2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_004/BraTS20_Training_004_seg.nii'
# img_nii = nib.load(nii_file_path)
#
# # 获取图像数据和空间信息
# img_data = img_nii.get_fdata()
# img_affine = img_nii.affine
#
# # 打印图像数据的形状
# print("Shape of NIfTI image data:", img_data.shape)
#
# # 可选：将数据转换为特定的数据类型
# img_data = np.asarray(img_data, dtype=np.float32)
# print(img_data)
# # 现在，img_data 是一个 NumPy 数组，可以在代码中使用
# import numpy as np
#
# # 读取单个 NumPy 数组（.npy 文件）
# file_path = 'D:\dataset\BRATS2020_Training_none_npy\seg\HG_BraTS20_Training_001_seg.npy'
# loaded_array = np.load(file_path)
#
# # 打印数组的形状
# print("Shape of the loaded array:", loaded_array.shape)
#
# # 现在，loaded_array 是一个 NumPy 数组，可以在代码中使用
# import math
# import torch
#
# def get_timestep_embedding(timesteps, embedding_dim):
#     """
#     This matches the implementation in Denoising Diffusion Probabilistic Models:
#     From Fairseq.
#     Build sinusoidal embeddings.
#     This matches the implementation in tensor2tensor, but differs slightly
#     from the description in Section 3.5 of "Attention Is All You Need".
#     """
#     assert len(timesteps.shape) == 1
#     half_dim = embedding_dim // 2
#     emb = math.log(10000) / (half_dim - 1)
#     emb = torch.exp(torch.arange(0, half_dim, dtype=torch.float32) * -emb)
#     emb = timesteps.float()[:, None] * emb[None, :]
#     emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
#     if embedding_dim % 2 == 1:  # zero pad
#         emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
#     return emb
# a = torch.tensor([1000])
# b = 128
#
# s = get_timestep_embedding(a, b)
# print(s)
# import torch
# a=torch.randn(3,5)
# num_val = len(a[0])
# for index in range(10):
#     print(index)


