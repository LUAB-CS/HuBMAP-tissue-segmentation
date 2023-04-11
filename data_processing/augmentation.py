import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import random as r

#Toutes les augmentations sont visualisables dans data_augmentation_check.ipynb

class RandomHorizontalFlip(object):

    def __call__(self, item):
        image, mask = item
        if r.random() > 0.5:
            return item
        image = image.flip(dims=(0,))
        mask = mask.flip(dims=(0,))
        return (image,mask)

class RandomVerticalFlip(object):

    def __call__(self, item):
        image, mask = item
        if r.random() > 0.5:
            return item
        image = image.flip(dims=(1,))
        mask = mask.flip(dims=(1,))
        return (image,mask)

class RandomRotation(object):

    def angle(self):
        #Sens de rotation
        return r.uniform(-180,180)

    def __call__(self,item):
        image,mask = item
        angle = self.angle()
        image = F.rotate(image,angle=angle)
        mask = F.rotate(mask,angle)
        return (image,mask)

class CustomColorJitter(object):

    def __init__(self,brightness = 0.3, hue = 0.3, saturation = 0.3):
        self.brightness = brightness
        self.hue = hue
        self.saturation = saturation

    def __call__(self,item):
        image, mask = item
        image = transforms.ColorJitter(brightness=self.brightness,hue=self.hue,saturation=self.saturation)(image)
        return (image,mask)

class RandomBlur(object):

    def __init__(self, kernel_size=25, blurred_ratio = 0.2):
        self.kernel_size = kernel_size
        self.blurred_ratio = blurred_ratio

    def __call__(self,item):
        image, mask = item
        if r.random() > self.blurred_ratio:
            return item
        image = F.gaussian_blur(image,kernel_size=self.kernel_size)
        return (image,mask)

class RandomCrop(object):

    def _init_(self, height = 512, width = 512,crop_ratio = 0.3):
        self.height = height
        self.width = width
        self.crop_ratio = crop_ratio
    
    def _call_(self,item):
        image, mask = item
        C,H,W = mask.shape
        if r.random() < self.crop_ratio:
            top = r.randint(0,H - self.height)
            left = r.randint(0, W - self.width)
            image_crop = F.crop(image,top,left,self.height,self.width)
            mask_crop = F.crop(mask,top,left,self.height,self.width)
            return (image_crop,mask_crop)
        return (image, mask)