import copy
import os
import random

import torch.utils.data
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms

def createdataset():
    data_dir = './dataset/'
    train_dir = data_dir + 'train'
    val_dir = data_dir + 'valid'
    train_list, val_list = [], []
    b_list = []
    for i in [train_dir, val_dir]:
        num_i = 0
        for a,b,c in os.walk(i):  #
            if(len(b) != 0):  # b为子文件夹名称
                b_list = copy.deepcopy(b)
            elif(len(c) != 0 and i == train_dir):
                for j in range(len(c)):  # c为文件名
                    train_data = os.path.join(train_dir,b_list[num_i],c[j]) + '\t' + str(int(b_list[num_i])-1) + '\n'
                    train_list.append(train_data)
                num_i = num_i + 1
            elif((len(c) != 0 and i == val_dir)):
                for j in range(len(c)):
                    val_data = os.path.join(val_dir,b_list[num_i],c[j]) + '\t' + str(int(b_list[num_i])-1) + '\n'
                    val_list.append(val_data)
                num_i = num_i + 1

    print(train_list)
    print('-----------------------')
    print(val_list)
    random.shuffle(train_list)
    random.shuffle(val_list)

    with open('./train.txt', 'w', encoding='UTF-8') as f:
        for train_image in train_list:
            f.write(str(train_image))
    with open('./valid.txt', 'w', encoding='UTF-8') as f:
        for val_image in val_list:
            f.write(str(val_image))

class LodaData(Dataset):
    def __init__(self, txt_path, train_flag = True):
        self.imgs_info = self.get_images(txt_path)
        self.train_flag = train_flag
        self.train_tf = transforms.Compose([transforms.Resize((224, 224)),  # 所用模型需要224的输入
                                transforms.RandomHorizontalFlip(p=0.5),  # 随机水平反转 概率0.5
                                transforms.RandomVerticalFlip(p=0.5),  # 随机竖直反转 概率0.5
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 均值 标准差
                                ])
        self.valid_tf = transforms.Compose([transforms.Resize((224, 224)),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                ])
    def get_images(self, txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            imgs_info = f.readlines()
            imgs_info = list(map(lambda x:x.strip().split('\t'), imgs_info))
            return imgs_info

    def padding_img(self, size, image):  # 加灰边
        w, h = size
        iw, ih = image.size

        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

        return new_image

    def __getitem__(self, item):
        img_path, label = self.imgs_info[item]
        img = Image.open(img_path)
        img = img.convert('RGB')
        # img = self.padding_img(img)
        if self.train_flag:
            img = self.train_tf(img)
        else:
            img = self.valid_tf(img)
        label = int(label)
        return img, label

    def __len__(self):
        return len(self.imgs_info)

if __name__ == "__main__":
    trainpath = './train.txt'
    valtrain = './valid.txt'
    if os.path.exists(trainpath) and os.path.exists(valtrain):
        print("文件存在")
    else:
        createdataset()

    # train_dataset = LodaData('train.txt', True)
    # train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=10, shuffle=True)
    # for image, label in train_loader:
    #     print(image.shape)
    #     print(image)
    #     print(label)

