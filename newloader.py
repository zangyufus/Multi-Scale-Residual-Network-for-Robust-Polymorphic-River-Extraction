import os
import torch
import rasterio
import numpy as np
import albumentations as A
import torchvision.transforms as transforms
import cv2
from torch.utils.data import Dataset
from rasterio.enums import Resampling

channel_means = [0.0] * 6
channel_stds = [1.0] * 6

class TifToTensor(object):
    def __call__(self, img):
        tf = transforms.Compose([transforms.ToTensor()])
        return tf(img)

# class TifDataset(Dataset):                    # 6

#     def __init__(self, img_dir, img_fnames, mask_dir=None, mask_fnames=None, target_size=(256, 256), isTrain=False, resize=True):
#         self.img_dir = img_dir
#         self.img_fnames = img_fnames
#         self.mask_dir = mask_dir
#         self.mask_fnames = mask_fnames
#         self.target_size = target_size  # (H, W)
#         self.isTrain = isTrain
#         self.resize = resize

#         self.aug = A.Compose([
#             A.HorizontalFlip(),
#             A.VerticalFlip(),
#             A.RandomRotate90(p=0.5)
#         ])

#     def __getitem__(self, i):
#         fname = self.img_fnames[i]
#         fpath = os.path.join(self.img_dir, fname)

#         # TIF
#         with rasterio.open(fpath) as dataset:
#             img = dataset.read([1, 2, 3, 4, 5, 6], resampling=Resampling.bilinear)
#             img = np.where(img == 255, 0, img)
#             img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)   #处理 NaN 和 Inf
#             height, width = dataset.height, dataset.width

#         # Z-score归一化
#         global channel_means, channel_stds
#         if not channel_means or not channel_stds:
#             all_pixels = img.reshape((6, -1))
#             channel_means = np.mean(all_pixels, axis=1)
#             channel_stds = np.std(all_pixels, axis=1)

#         # 2%线性拉伸
#         img_stretched = np.zeros_like(img, dtype=np.float32)
#         img_stretched = np.nan_to_num(img_stretched, nan=0.0, posinf=0.0, neginf=0.0)
#         for b in range(6):
#             lower_percentile = np.percentile(img[b], 2)
#             upper_percentile = np.percentile(img[b], 98)
#             img_stretched[b] = np.clip(img[b], lower_percentile, upper_percentile)
#             img_stretched[b] = (img_stretched[b] - lower_percentile) / (upper_percentile - lower_percentile)

#         img_resized = np.zeros((6, self.target_size[0], self.target_size[1]), dtype=np.float32)
#         for b in range(6):
#             if self.resize:
#                 img_resized[b] = cv2.resize(img_stretched[b], (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_CUBIC)
#             else:
#                 img_resized[b] = img_stretched[b]

#         img_resized = np.moveaxis(img_resized, 0, -1)

#         # mask
#         mask_resized = None
#         if self.mask_dir is not None and self.mask_fnames is not None:
#             mask_fname = self.mask_fnames[i]
#             mask_fpath = os.path.join(self.mask_dir, mask_fname)

#             with rasterio.open(mask_fpath) as mask_dataset:
#                 mask = mask_dataset.read(1, resampling=Resampling.nearest)

#                 if self.resize:
#                     mask_resized = cv2.resize(mask.astype(np.uint8), (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_NEAREST)
#                 else:
#                     mask_resized = mask

#             mask_resized = mask_resized.astype(np.uint8)
#             mask_resized = np.where(mask_resized == 255, 0, mask_resized)
#             mask_resized = np.expand_dims(mask_resized, axis=-1)
#             mask_resized = np.repeat(mask_resized, 6, axis=-1)

#         # print(f"Image size: {img_resized.shape}")
#         # print(f"Mask size: {mask_resized.shape}")


#         # 随机变换
#         augmented = self.aug(image=img_resized, mask=mask_resized)
#         if self.isTrain:
#             augmented = self.aug(image=img_resized, mask=mask_resized)
#             img_resized = augmented['image']
#             mask_resized = augmented['mask']

#         mask_single_channel = np.expand_dims(mask_resized[:, :, 0], axis=0)

#         # 转换为 PyTorch Tensor 格式
#         img_resized = np.nan_to_num(img_resized, nan=0.0, posinf=0.0, neginf=0.0)

#         img_resized = torch.tensor(img_resized, dtype=torch.float).permute(2, 0, 1) 

#         if np.any(np.isnan(mask_single_channel)) or np.any(np.isinf(mask_single_channel)):
#             mask_single_channel = np.nan_to_num(mask_single_channel, nan=0, posinf=0, neginf=0)
#         if mask_resized is not None:
#             mask_single_channel = torch.tensor(mask_single_channel, dtype=torch.long)  # (1, H, W)

#         # print(f"Final image shape: {img_resized.shape}")  # (6, 256, 256)
#         # print(f"Mask shape: {mask_resized.shape if mask_resized is not None else 'None'}")  # (1, 256, 256)
#         # print(f"Min image value: {img_resized.min()}, Max image value: {img_resized.max()}")
#         # print(f"Min mask value: {mask_resized.min()}, Max mask value: {mask_resized.max()}")

#         return {
#             'image': img_resized,  # (C, H, W)
#             'mask': mask_single_channel,  # (1, H, W)
#             'original_size': torch.tensor((height, width), dtype=torch.int),
#             'resized_size': torch.tensor(self.target_size, dtype=torch.int),
#             'img_path': fpath
#         }

#     def __len__(self):
#         return len(self.img_fnames)

class TifDataset(Dataset):    #************************************************************************6+5+5
    def __init__(self, img_dir, img_fnames, mask_dir=None, mask_fnames=None, target_size=(256, 256), isTrain=False, resize=True):
        self.img_dir = img_dir
        self.img_fnames = img_fnames
        self.mask_dir = mask_dir
        self.mask_fnames = mask_fnames
        self.target_size = target_size  # (H, W)
        self.isTrain = isTrain
        self.resize = resize
        self.aug = A.Compose([
                #A.augmentations.MotionBlur(p=0.1),  # 运动模糊
                A.HorizontalFlip(p=0.5),  # 水平翻转
                A.VerticalFlip(p=0.5),  # 垂直翻转
                A.augmentations.geometric.rotate.RandomRotate90(p=0.5),  # 随机旋转90度
                A.RandomBrightnessContrast(p=0.5),  # 随机调整亮度与对比度                
                #A.RandomGamma(p=0.5)  # 随机Gamma校正
            ])

    def __getitem__(self, i):
        fname = self.img_fnames[i]
        fpath = os.path.join(self.img_dir, fname)

        # TIF
        with rasterio.open(fpath) as dataset:
            img = dataset.read([1, 2, 3, 4, 5, 6], resampling=Resampling.bilinear)
            img = np.where(img == 255, 0, img)
            img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)   #处理 NaN 和 Inf
            height, width = dataset.height, dataset.width

        # 计算 NDWI 和 MNDWI
        # 0-R 1-G 2-B 3-NIR 4-SWIR1 5-SWIR2
        # NDWI = (Green – NIR) / (Green + NIR) 
        # MNDWI = (Green – SWIR1) / (Green + SWIR1) 
        # SWI = (NIR – SWIR1) / (NIR + SWIR1) 
        # MBWI = Blue + Green – NIR – SWIR1

        ndwi = (img[1] - img[3]) / (img[1] + img[3] + 1e-6)
        mndwi1 = (img[1] - img[4]) / (img[1] + img[4] + 1e-6) 
        mndwi2 = (img[1] - img[5]) / (img[1] + img[5] + 1e-6) 
        swi = (img[3] - img[4]) / (img[3] + img[4] + 1e-6) 
        mbwi = img[2] + img[1] - img[3] - img[4] + 1e-6
        wi2015 = 1.7204 + 171*img[1] + 3*img[2] - 70*img[3] - 45*img[4] - 71*img[5]

        # 2%线性拉伸
        img_stretched = np.zeros_like(img, dtype=np.float32)
        img_stretched = np.nan_to_num(img_stretched, nan=0.0, posinf=0.0, neginf=0.0)
        for b in range(6):
            lower_percentile = np.percentile(img[b], 2)
            upper_percentile = np.percentile(img[b], 98)
            img_stretched[b] = np.clip(img[b], lower_percentile, upper_percentile)
            img_stretched[b] = (img_stretched[b] - lower_percentile) / (upper_percentile - lower_percentile)

        img_resized = np.zeros((6, self.target_size[0], self.target_size[1]), dtype=np.float32)

        for b in range(6):
            if self.resize:
                img_resized[b] = cv2.resize(img_stretched[b], (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_CUBIC)
            else:
                img_resized[b] = img_stretched[b]

        ndwi_resized = cv2.resize(ndwi, self.target_size, interpolation=cv2.INTER_CUBIC)
        mndwi1_resized = cv2.resize(mndwi1, self.target_size, interpolation=cv2.INTER_CUBIC)
        mndwi2_resized = cv2.resize(mndwi2, self.target_size, interpolation=cv2.INTER_CUBIC)
        swi_resized = cv2.resize(swi, self.target_size, interpolation=cv2.INTER_CUBIC)
        mbwi_resized = cv2.resize(mbwi, self.target_size, interpolation=cv2.INTER_CUBIC)
        wi2015_resized = cv2.resize(wi2015, self.target_size, interpolation=cv2.INTER_CUBIC)

        # print(f"Original image size: {img_resized.shape}")
        # print(f"NDWI resized size: {ndwi_resized.shape}")
        # print(f"MNDWI resized size: {mndwi_resized.shape}")
        # 扩展 NDWI 和 MNDWI 为 6 通道
        ndwi_resized = np.repeat(np.expand_dims(ndwi_resized, axis=0), 2, axis=0)
        mndwi1_resized = np.repeat(np.expand_dims(mndwi1_resized, axis=0), 2, axis=0)
        mndwi2_resized = np.repeat(np.expand_dims(mndwi2_resized, axis=0), 2, axis=0)
        swi_resized = np.repeat(np.expand_dims(swi_resized, axis=0), 2, axis=0)
        mbwi_resized = np.repeat(np.expand_dims(mbwi_resized, axis=0), 2, axis=0)
        wi2015_resized = np.repeat(np.expand_dims(wi2015_resized, axis=0), 2, axis=0)
        # print(f"NDWI resized size: {ndwi_resized.shape}")
        # print(f"MNDWI resized size: {mndwi_resized.shape}")

        # ndwi_resized = np.expand_dims(ndwi_resized, axis=0)  # 变成 (H, W, 1)
        # mndwi_resized = np.expand_dims(mndwi_resized, axis=0)  # 变成 (H, W, 1)
        # # print(f"NDWI resized size: {ndwi_resized.shape}")
        # # print(f"MNDWI resized size: {mndwi_resized.shape}")

        # 将原始图像和重复的 NDWI 和 MNDWI 拼接在一起
        img_resized = np.concatenate([img_resized, ndwi_resized, mndwi1_resized, mndwi2_resized, wi2015_resized, mbwi_resized], axis=0)
        # img_resized = np.concatenate([img_resized, ndwi_resized, mndwi1_resized], axis=0)
        # print(f"Concatenated image size: {img_resized.shape}")

        # 转换通道顺序 (H, W, C) -> (C, H, W)
        img_resized = np.moveaxis(img_resized, 0, -1)

        # mask
        mask_resized = None
        if self.mask_dir is not None and self.mask_fnames is not None:
            mask_fname = self.mask_fnames[i]
            mask_fpath = os.path.join(self.mask_dir, mask_fname)

            with rasterio.open(mask_fpath) as mask_dataset:
                mask = mask_dataset.read(1, resampling=Resampling.nearest)

                if self.resize:
                    mask_resized = cv2.resize(mask.astype(np.uint8), (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_NEAREST)
                else:
                    mask_resized = mask

            mask_resized = mask_resized.astype(np.uint8)
            mask_resized = np.where(mask_resized == 255, 0, mask_resized)
            mask_resized = np.expand_dims(mask_resized, axis=-1)
            mask_resized = np.repeat(mask_resized, 6, axis=-1)

        # 随机变换
        augmented = self.aug(image=img_resized, mask=mask_resized)
        if self.isTrain:
            img_resized = augmented['image']
            mask_resized = augmented['mask']

        mask_single_channel = np.expand_dims(mask_resized[:, :, 0], axis=0)

        # 转换为 PyTorch Tensor 格式
        img_resized = np.nan_to_num(img_resized, nan=0.0, posinf=0.0, neginf=0.0)
        img_resized = torch.tensor(img_resized, dtype=torch.float).permute(2, 0, 1) 

        if np.any(np.isnan(mask_single_channel)) or np.any(np.isinf(mask_single_channel)):
            mask_single_channel = np.nan_to_num(mask_single_channel, nan=0, posinf=0, neginf=0)

        if mask_resized is not None:
            mask_single_channel = torch.tensor(mask_single_channel, dtype=torch.long)  # (1, H, W)

        return {
            'image': img_resized,  # (C, H, W)
            'mask': mask_single_channel,  # (1, H, W)
            'original_size': torch.tensor((height, width), dtype=torch.int),
            'resized_size': torch.tensor(self.target_size, dtype=torch.int),
            'img_path': fpath,
            'fname':fname
        }

    def __len__(self):
        return len(self.img_fnames)




# class TifDataset(Dataset):                        # 3(rgb)

#     def __init__(self, img_dir, img_fnames, mask_dir=None, mask_fnames=None, target_size=(256, 256), isTrain=False, resize=True):
#         self.img_dir = img_dir
#         self.img_fnames = img_fnames
#         self.mask_dir = mask_dir
#         self.mask_fnames = mask_fnames
#         self.target_size = target_size  # (H, W)
#         self.isTrain = isTrain
#         self.resize = resize

#         self.aug = A.Compose([
#             A.HorizontalFlip(),
#             A.VerticalFlip(),
#             A.RandomRotate90(p=0.5)
#         ])

#     def __getitem__(self, i):
#         fname = self.img_fnames[i]
#         fpath = os.path.join(self.img_dir, fname)

#         # TIF
#         with rasterio.open(fpath) as dataset:
#             img = dataset.read([1, 2, 3], resampling=Resampling.bilinear)  # 仅读取前三个波段
#             img = np.where(img == 255, 0, img)
#             img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
#             height, width = dataset.height, dataset.width

#         # 2%线性拉伸
#         img_stretched = np.zeros_like(img, dtype=np.float32)
#         for b in range(3):
#             lower_percentile = np.percentile(img[b], 2)
#             upper_percentile = np.percentile(img[b], 98)
#             img_stretched[b] = np.clip(img[b], lower_percentile, upper_percentile)
#             img_stretched[b] = (img_stretched[b] - lower_percentile) / (upper_percentile - lower_percentile)

#         img_resized = np.zeros((3, self.target_size[0], self.target_size[1]), dtype=np.float32)
#         for b in range(3):
#             if self.resize:
#                 img_resized[b] = cv2.resize(img_stretched[b], (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_CUBIC)
#             else:
#                 img_resized[b] = img_stretched[b]

#         img_resized = np.moveaxis(img_resized, 0, -1)

#         # mask
#         mask_resized = None
#         if self.mask_dir is not None and self.mask_fnames is not None:
#             mask_fname = self.mask_fnames[i]
#             mask_fpath = os.path.join(self.mask_dir, mask_fname)

#             with rasterio.open(mask_fpath) as mask_dataset:
#                 mask = mask_dataset.read(1, resampling=Resampling.nearest)

#                 if self.resize:
#                     mask_resized = cv2.resize(mask.astype(np.uint8), (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_NEAREST)
#                 else:
#                     mask_resized = mask

#             mask_resized = mask_resized.astype(np.uint8)
#             mask_resized = np.where(mask_resized == 255, 0, mask_resized)
#             mask_resized = np.expand_dims(mask_resized, axis=-1)

#         # 随机变换
#         if self.isTrain:
#             augmented = self.aug(image=img_resized, mask=mask_resized)
#             img_resized = augmented['image']
#             mask_resized = augmented['mask']

#         mask_single_channel = np.expand_dims(mask_resized[:, :, 0], axis=0) if mask_resized is not None else None

#         # 转换为 PyTorch Tensor 格式
#         img_resized = np.nan_to_num(img_resized, nan=0.0, posinf=0.0, neginf=0.0)
#         img_resized = torch.tensor(img_resized, dtype=torch.float).permute(2, 0, 1)  # (3, H, W)

#         if mask_single_channel is not None:
#             mask_single_channel = torch.tensor(mask_single_channel, dtype=torch.long)  # (1, H, W)

#         return {
#             'image': img_resized,  # (3, H, W)
#             'mask': mask_single_channel,  # (1, H, W) or None
#             'original_size': torch.tensor((height, width), dtype=torch.int),
#             'resized_size': torch.tensor(self.target_size, dtype=torch.int),
#             'img_path': fpath,
#             'fname':fname
#         }

#     def __len__(self):
#         return len(self.img_fnames)





# class TifDataset(Dataset):
#     def __init__(self, img_dir, img_fnames, mask_dir=None, mask_fnames=None, target_size=(256, 256), isTrain=False, resize=True):
#         self.img_dir = img_dir
#         self.img_fnames = img_fnames
#         self.mask_dir = mask_dir
#         self.mask_fnames = mask_fnames
#         self.target_size = target_size  # (H, W)
#         self.isTrain = isTrain
#         self.resize = resize
#         self.aug = A.Compose([
#                 # 对所有波段进行相同变换
#                 A.augmentations.crops.transforms.RandomResizedCrop(256, 256, p=0.5),  # 随机裁剪并缩放
#                 A.augmentations.MotionBlur(p=0.1),  # 运动模糊
#                 A.HorizontalFlip(p=0.5),  # 水平翻转
#                 A.VerticalFlip(p=0.5),  # 垂直翻转
#                 A.augmentations.geometric.rotate.RandomRotate90(p=0.5),  # 随机旋转90度
                
#                 # 对所有波段应用亮度和对比度调整
#                 A.RandomBrightnessContrast(p=0.5),  # 随机调整亮度与对比度
                
#                 # 对所有波段应用Gamma变换
#                 A.RandomGamma(p=0.5)  # 随机Gamma校正
#             ])

#     def __getitem__(self, i):
#         fname = self.img_fnames[i]
#         fpath = os.path.join(self.img_dir, fname)

#         # TIF
#         with rasterio.open(fpath) as dataset:
#             img = dataset.read([1, 2, 3, 4, 5, 6], resampling=Resampling.bilinear)
#             img = np.where(img == 255, 0, img)
#             img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)   # 处理 NaN 和 Inf
#             height, width = dataset.height, dataset.width

#         # 计算 NDWI 和 MNDWI
#         ndwi = (img[1] - img[3]) / (img[1] + img[3] + 1e-6)  # 假设红色通道是索引1，近红外是索引3
#         mndwi = (img[1] - img[4]) / (img[1] + img[4] + 1e-6)  # 假设绿色通道是索引2，近红外是索引4，SWIR是索引5

#         # 2%线性拉伸
#         img_stretched = np.zeros_like(img, dtype=np.float32)
#         img_stretched = np.nan_to_num(img_stretched, nan=0.0, posinf=0.0, neginf=0.0)
#         for b in range(6):
#             lower_percentile = np.percentile(img[b], 2)
#             upper_percentile = np.percentile(img[b], 98)
#             img_stretched[b] = np.clip(img[b], lower_percentile, upper_percentile)
#             img_stretched[b] = (img_stretched[b] - lower_percentile) / (upper_percentile - lower_percentile)

#         print(f"Original image size: {img_stretched.shape}")

#         # 将 NDWI 和 MNDWI 扩展成通道，准备增强
#         ndwi_resized = np.expand_dims(ndwi, axis=0)  # 变成 (H, W, 1)
#         mndwi_resized = np.expand_dims(mndwi, axis=0)  # 变成 (H, W, 1)
        
#         # 将所有通道拼接在一起以供增强
#         img_combined = np.concatenate([img_stretched, ndwi_resized, mndwi_resized], axis=0)  # (H, W, 8)
#         img_combined = np.moveaxis(img_combined, 0, -1)
#         print(f"Combined image size: {img_combined.shape}")

#         # Mask
#         mask_resized = None
#         if self.mask_dir is not None and self.mask_fnames is not None:
#             mask_fname = self.mask_fnames[i]
#             mask_fpath = os.path.join(self.mask_dir, mask_fname)

#             with rasterio.open(mask_fpath) as mask_dataset:
#                 mask = mask_dataset.read(1, resampling=Resampling.nearest)
#                 mask_resized = mask.astype(np.uint8)
#                 mask_resized = np.where(mask_resized == 255, 0, mask_resized)
#                 mask_resized = np.expand_dims(mask_resized, axis=-1)  # 扩展维度 (H, W, 1)

#         # 随机变换增强
#         augmented = self.aug(image=img_combined, mask=mask_resized)
#         img_resized = augmented['image']
#         mask_resized = augmented['mask']
#         print(f"Augmented image size: {img_resized.shape}")
#         print(f"Augmented mask size: {mask_resized.shape}")

#         # 处理增强后的掩码
#         mask_single_channel = np.expand_dims(mask_resized[:, :, 0], axis=0)  # (1, H, W)

#         # 转换为 PyTorch Tensor 格式
#         img_resized = np.nan_to_num(img_resized, nan=0.0, posinf=0.0, neginf=0.0)
#         img_resized = torch.tensor(img_resized, dtype=torch.float).permute(2, 0, 1)  # (C, H, W)

#         if np.any(np.isnan(mask_single_channel)) or np.any(np.isinf(mask_single_channel)):
#             mask_single_channel = np.nan_to_num(mask_single_channel, nan=0, posinf=0, neginf=0)

#         if mask_resized is not None:
#             mask_single_channel = torch.tensor(mask_single_channel, dtype=torch.long)  # (1, H, W)

#         return {
#             'image': img_resized,  # (C, H, W)
#             'mask': mask_single_channel,  # (1, H, W)
#             'ndwi': ndwi_resized,  # (H, W, 1)
#             'mndwi': mndwi_resized,  # (H, W, 1)
#             'original_size': torch.tensor((height, width), dtype=torch.int),
#             'resized_size': torch.tensor(self.target_size, dtype=torch.int),
#             'img_path': fpath
#         }

#     def __len__(self):
#         return len(self.img_fnames)






# class TifDataset(Dataset):          # 变换，八通道
#     def __init__(self, img_dir, img_fnames, mask_dir=None, mask_fnames=None, target_size=(256, 256), isTrain=False, resize=True):
#         self.img_dir = img_dir
#         self.img_fnames = img_fnames
#         self.mask_dir = mask_dir
#         self.mask_fnames = mask_fnames
#         self.target_size = target_size  # (H, W)
#         self.isTrain = isTrain
#         self.resize = resize
#         self.aug = A.Compose([
#                 # 对所有波段进行相同变换
#                 #A.augmentations.MotionBlur(p=0.1),  # 运动模糊
#                 A.HorizontalFlip(p=0.5),  # 水平翻转
#                 A.VerticalFlip(p=0.5),  # 垂直翻转
#                 A.augmentations.geometric.rotate.RandomRotate90(p=0.5),  # 随机旋转90度
#                 #A.RandomBrightnessContrast(p=0.5),  # 随机调整亮度与对比度                
#                 A.RandomGamma(p=0.5)  # 随机Gamma校正
#             ])

#     def __getitem__(self, i):
#         fname = self.img_fnames[i]
#         fpath = os.path.join(self.img_dir, fname)

#         # TIF
#         with rasterio.open(fpath) as dataset:
#             img = dataset.read([1, 2, 3, 4, 5, 6], resampling=Resampling.bilinear)
#             img = np.where(img == 255, 0, img)
#             img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)   #处理 NaN 和 Inf
#             height, width = dataset.height, dataset.width

#         # 2%线性拉伸
#         img_stretched = np.zeros_like(img, dtype=np.float32)
#         img_stretched = np.nan_to_num(img_stretched, nan=0.0, posinf=0.0, neginf=0.0)
#         for b in range(6):
#             lower_percentile = np.percentile(img[b], 2)
#             upper_percentile = np.percentile(img[b], 98)
#             img_stretched[b] = np.clip(img[b], lower_percentile, upper_percentile)
#             img_stretched[b] = (img_stretched[b] - lower_percentile) / (upper_percentile - lower_percentile)

#         img_resized = np.zeros((6, self.target_size[0], self.target_size[1]), dtype=np.float32)
#         for b in range(6):
#             if self.resize:
#                 img_resized[b] = cv2.resize(img_stretched[b], (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_CUBIC)
#             else:
#                 img_resized[b] = img_stretched[b]

#         # 计算 NDWI 和 MNDWI
#         ndwi = (img_resized[1] - img_resized[3]) / (img_resized[1] + img_resized[3] + 1e-6)  # 假设红色通道是索引1，近红外是索引3
#         mndwi = (img_resized[1] - img_resized[4]) / (img_resized[1] + img_resized[4] + 1e-6)  # 假设绿色通道是索引2，近红外是索引4，SWIR是索引5
        
#         #将 NDWI 和 MNDWI 扩展成通道，准备增强
#         ndwi_resized = np.expand_dims(ndwi, axis=0)  # 变成 (H, W, 1)
#         mndwi_resized = np.expand_dims(mndwi, axis=0)  # 变成 (H, W, 1)
#         # print(f"NDWI image size: {ndwi_resized.shape}")
#         # print(f"MNDWI image size: {mndwi_resized.shape}")
        
#         # 将所有通道拼接在一起以供增强
#         img_combined = np.concatenate([img_resized, ndwi_resized, mndwi_resized], axis=0)  # (H, W, 8)
#         # print(f"Combined image size: {img_combined.shape}")

#         img_resized = np.moveaxis(img_combined, 0, -1)

#         # mask
#         mask_resized = None
#         if self.mask_dir is not None and self.mask_fnames is not None:
#             mask_fname = self.mask_fnames[i]
#             mask_fpath = os.path.join(self.mask_dir, mask_fname)

#             with rasterio.open(mask_fpath) as mask_dataset:
#                 mask = mask_dataset.read(1, resampling=Resampling.nearest)

#                 if self.resize:
#                     mask_resized = cv2.resize(mask.astype(np.uint8), (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_NEAREST)
#                 else:
#                     mask_resized = mask

#             mask_resized = mask_resized.astype(np.uint8)
#             mask_resized = np.where(mask_resized == 255, 0, mask_resized)
#             mask_resized = np.expand_dims(mask_resized, axis=-1)
#             mask_resized = np.repeat(mask_resized, 6, axis=-1)

#         # 随机变换
#         augmented = self.aug(image=img_resized, mask=mask_resized)
#         if self.isTrain:
#             img_resized = augmented['image']
#             mask_resized = augmented['mask']

#         mask_single_channel = np.expand_dims(mask_resized[:, :, 0], axis=0)

#         # 转换为 PyTorch Tensor 格式
#         img_resized = np.nan_to_num(img_resized, nan=0.0, posinf=0.0, neginf=0.0)
#         img_resized = torch.tensor(img_resized, dtype=torch.float).permute(2, 0, 1) 

#         if np.any(np.isnan(mask_single_channel)) or np.any(np.isinf(mask_single_channel)):
#             mask_single_channel = np.nan_to_num(mask_single_channel, nan=0, posinf=0, neginf=0)

#         if mask_resized is not None:
#             mask_single_channel = torch.tensor(mask_single_channel, dtype=torch.long)  # (1, H, W)

#         return {
#             'image': img_resized,  # (C, H, W)
#             'mask': mask_single_channel,  # (1, H, W)
#             'original_size': torch.tensor((height, width), dtype=torch.int),
#             'resized_size': torch.tensor(self.target_size, dtype=torch.int),
#             'img_path': fpath
#         }

#     def __len__(self):
#         return len(self.img_fnames)