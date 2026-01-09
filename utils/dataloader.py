import numpy as np
from PIL import Image, ImageEnhance
from torchvision import datasets, transforms
import torch
import random


class DataAugmentation:
    def __init__(self):
        pass

    @staticmethod
    def openImage(image,label,size=128):
        image = Image.open(image, mode="r")
        if label is not None:
            label=Image.open(label, mode="r")
        return image,label

    @staticmethod
    def resizeImage(image, label,size=128):
        size=int(1.05*size)
        image=image.resize((size, size))
        if label is not None:
            label=label.resize((size,size), Image.ANTIALIAS)
        return image,label

    @staticmethod
    def randomRotation(image, label, mode=Image.BICUBIC,size=128):

        random_angle = np.random.randint(1, 360)
        if label is not None:
            return image.rotate(random_angle, mode), label.rotate(random_angle, Image.NEAREST)
        else:
            return image.rotate(random_angle, mode), label

    # 暂时未使用这个函数
    @staticmethod
    def randomCrop(image, label,size=128):

        image_width = image.size[0]
        image_height = image.size[1]
        crop_win_size = np.random.randint(size-18, size)
        random_region = (
            (image_width - crop_win_size) >> 1, (image_height - crop_win_size) >> 1, (image_width + crop_win_size) >> 1,
            (image_height + crop_win_size) >> 1)
        image = image.crop(random_region)
        if label is not None:
            label=label.crop(random_region)
        return image, label

    @staticmethod
    def randomColor(image, label,size=128):

        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
        random_factor = np.random.randint(10, 21) / 10.  # 随机因子
        brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
        random_factor = np.random.randint(10, 21) / 10.  # 随机因1子
        contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        return ImageEnhance.Sharpness(contrast_image).enhance(random_factor), label  # 调整图像锐度

    @staticmethod
    def randomGaussian(image, label, mean=0.2, sigma=0.3,size=None):


        def gaussianNoisy(im, mean=0.2, sigma=0.3):

            for _i in range(len(im)):
                im[_i] += random.gauss(mean, sigma)
            return im

        # 将图像转化成数组
        img = np.asarray(image)
        img.flags.writeable = True  # 将数组改为读写模式
        width, height = img.shape[:2]
        img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, sigma)
        img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)
        img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)
        img[:, :, 0] = img_r.reshape([width, height])
        img[:, :, 1] = img_g.reshape([width, height])
        img[:, :, 2] = img_b.reshape([width, height])
        return Image.fromarray(np.uint8(img)), label

    @staticmethod
    def array2tensor(image,label,size=128):
        image=Image.fromarray(np.uint8(image))
        if label is not None:
            label=Image.fromarray(np.uint8(label))
        return image,label

    @staticmethod
    def saveImage(image, path):
        image.save(path)



class DATASET():
    def __init__(self,dataset,path,root, input_size=128,is_train=True):
        self.dataset=dataset
        self.input_size = input_size
        self.path=path
        self.is_train=is_train
        self.data=self.readPath(path)
        self.root=root
        self.seed=0
        self.Aug=DataAugmentation
        self.transform_data = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor()])

    def readPath(self,path):
        data=[]
        for line in open(path,encoding='UTF'):
            line=line.replace('\n','')
            data.append(line)
        return data

    def traindataAug(self,image,label,size):
        image,label=self.Aug.openImage(image,label,size=size)
        image,label=self.Aug.resizeImage(image,label,size=size)
        image,label=self.Aug.randomRotation(image,label,size=size)
        image,label=self.Aug.randomCrop(image,label,size=size)
        image,label=self.Aug.array2tensor(image,label,size=size)
        return image,label

    def testdataAug(self,image,label,size):
        image, label = self.Aug.openImage(image, label,size=size)
        image, label = self.Aug.resizeImage(image, label,size=size)
        image, label = self.Aug.array2tensor(image, label,size=size)
        return image,label

    def __getitem__(self, index):
        img,target=(self.data[index]).split('[token]')

        target=int(target)
        imgData=self.root+'/'+img
        name=imgData
        mask=None
        if self.is_train==True:
            imgData, mask = self.traindataAug(imgData, mask, size=self.input_size)
            imgData = self.transform_data(imgData)
            # mask=self.transform_val(mask)
            if imgData.shape[0] > 3:
                imgData = torch.mean(imgData, dim=0, keepdim=True)
            if imgData.shape[0] < 3:
                imgData = imgData.repeat(3, 1, 1)
        else:
            imgData, mask = self.testdataAug(imgData, mask, size=self.input_size)
            imgData = self.transform_data(imgData)
            # mask=self.transform_val(mask)
            if imgData.shape[0] > 3:
                imgData = torch.mean(imgData, dim=0, keepdim=True)
            if imgData.shape[0] < 3:
                imgData = imgData.repeat(3, 1, 1)
        return index,imgData,target,name

    def __len__(self):
        return len(self.data)
