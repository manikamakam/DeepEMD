import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os
import numpy as np

class Custom(Dataset):

    def __init__(self, setname, args):

        if setname == 'train':
            IMAGE_PATH = os.path.join(args.data_dir, 'images')
            data = []
            label = []
            list = os.listdir(IMAGE_PATH)
            for dat in list:
                path = osp.join(IMAGE_PATH, dat)
                lb = dat.split(".")
                data.append(path)
                label.append(lb[0])

            self.data = data  # data path of all data
            self.label = label  # label of all data
            self.num_class = len(set(label))

            image_size = 500
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        if setname == 'val':
            IMAGE_PATH = os.path.join(args.data_dir, 'val')
            data = []
            label = []
            list = os.listdir(IMAGE_PATH)
            for dat in list:
                path = osp.join(IMAGE_PATH, dat)
                lb = dat.split(".")
                data.append(path)
                # print(lb[0])
                label.append(lb[0])

            self.data = data  # data path of all data
            self.label = label  # label of all data
            self.num_class = len(set(label))

            image_size = 500
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])

        if setname == 'test':
            IMAGE_PATH = os.path.join(args.data_dir, 'img_patches_filtered')
            data = []
            label = []
            list = os.listdir(IMAGE_PATH)
            for dat in list:
                path = osp.join(IMAGE_PATH, dat)
                data.append(path)
                label.append(0)

            self.data = data  # data path of all data
            self.label = label  # label of all data

            image_size = 500
            self.transform = transforms.Compose([
                transforms.Resize([550, 550]),
                transforms.CenterCrop(image_size),

                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label


if __name__ == '__main__':
    pass