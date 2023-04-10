import torchvision.transforms as transforms
import os, random, torch, glob
import numpy as np
import imgaug.augmenters as iaa

from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm
from PIL import Image

class jpeg_compression():

    def __init__(self):
        self.jc = iaa.JpegCompression(compression=(10, 50))

    def __call__(self,inp):
        return Image.fromarray(self.jc(image=np.array(inp)))

AugmentationGeneratorDict = {
                            "sharpness" : transforms.RandomAdjustSharpness(sharpness_factor=2), 
                            "Invert" : transforms.RandomInvert(),
                            "Solarize" : transforms.RandomSolarize(threshold=192.0),
                            "Equalize" : transforms.RandomEqualize(),
                            "Posterize" : transforms.RandomPosterize(bits=2),
                            "Random perspective" : transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
                            "Hflip" : transforms.RandomHorizontalFlip(p=0.5),
                            "Vflip" : transforms.RandomVerticalFlip(p=0.5),
                            "ColorJitter" : transforms.ColorJitter(brightness=0.5,contrast=0.5,hue=0.5),
                            "JpegCompresion" : jpeg_compression(),
                            }

PrepareData = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])

class CustomDataSet():
    def __init__(self, data_path, img_size):
        subdirs = sorted(os.listdir(data_path))
        self.num_classes = len(subdirs)
        self.img_size = img_size
        self.data = []
        for label,subdir in enumerate(subdirs):
            imgs = glob.glob(os.path.join(data_path,subdir,"*.jpg"))
            imgs.extend(glob.glob(os.path.join(data_path,subdir,"*.png")))
            imgs.extend(glob.glob(os.path.join(data_path,subdir,"*.jpeg")))
            imgs = [(img,label) for img in imgs]
            self.data += imgs

    def __getitem__(self,index):
        im = self.data[index][0]
        im = Image.open(im).resize((self.img_size,self.img_size))
        if len(np.array(im).shape) == 2:
            im = Image.fromarray(np.stack((np.array(im),)*3, axis=-1))
        im = AugmentationGeneratorDict[list(AugmentationGeneratorDict)[random.randint(0, len(AugmentationGeneratorDict)-1)]](im.copy())
        targ = self.data[index][1]
        return PrepareData(im) , torch.LongTensor([targ])

    def balancer(self):
        print("computing balanced weights ...")
        targ_count = [0] * self.num_classes
        for d in tqdm(self.data):
            targ = d[1]
            targ_count[targ] += 1
        all_data = len(self.data)
        targ_weight = list(all_data/np.array(targ_count))
        print("Balanced weights : " ,targ_weight)
        sample_weights = [0] * all_data
        for idx , d in enumerate(tqdm(self.data)):
            targ = d[1]
            sample_weights[idx] = targ_weight[targ]
        return WeightedRandomSampler(sample_weights,num_samples = len(sample_weights),replacement=True)

    def __len__(self):
        return len(self.data)

def ToCUDA(model):
    if torch.cuda.is_available():
        return model.cuda()
    else:
        return model

def clone_weights(model):
    weights = []
    for param in model.parameters():
        weights.append(param.clone())
    return weights

def weights_checker(w1,w2):
    if not len(w1) == len(w2):
        return "Weights are not same size , model has changed !"
    c = 0
    for i in zip(w1, w2):
        if torch.equal(i[0], i[1]):
            c+=1
    if not c == 0:
        return "{} of {} not updating !".format(c,len(w1))
    return "All weights are updating ..."

def SanityCheck(Model):
    for param in Model.parameters(): # check later (not a priority !)
        if param.requires_grad == False:
            print(param)