from __future__ import print_function
import torch
import numpy as np
import linecache
import torch.utils.data as data
from torchvision.datasets.folder import default_loader


class ES_Imagenet_Official(data.Dataset):
    def __init__(self, mode, data_set_path='../'):
        super().__init__()
        self.mode = mode
        self.filenames = []
        self.trainpath = data_set_path+'train'
        self.testpath = data_set_path+'val'
        self.traininfotxt = data_set_path+'trainlabel.txt'
        self.testinfotxt = data_set_path+'vallabel.txt'
        self.formats = '.npz'
        if mode == 'train':
            self.path = self.trainpath
            trainfile = open(self.traininfotxt, 'r')
            for line in trainfile:
                filename, classnum, a, b = line.split()
                realname,sub = filename.split('.')
                self.filenames.append(realname+self.formats)
        else:
            self.path = self.testpath
            testfile = open(self.testinfotxt, 'r')
            for line in testfile:
                filename, classnum, a, b = line.split()
                realname,sub = filename.split('.')
                self.filenames.append(realname+self.formats)

    def __getitem__(self, index):
        if self.mode == 'train':
            info = linecache.getline(self.traininfotxt, index+1)
        else:
            info = linecache.getline(self.testinfotxt, index+1)
        filename, classnum, a, b = info.split()
        realname,sub = filename.split('.')
        filename = realname+self.formats
        filename = self.path + r'/' + filename
        classnum = int(classnum)
        a = int(a)
        b = int(b)
        datapos = np.load(filename)['pos'].astype(np.float64)
        dataneg = np.load(filename)['neg'].astype(np.float64)
        
        dy = (254 - b) // 2
        dx = (254 - a) // 2
        input = torch.zeros([8, 2, 256, 256])

        x = datapos[:,0]+ dx
        y = datapos[:,1]+ dy
        t = datapos[:,2]-1
        input[t ,0,x ,y ] = 1
        
        x = dataneg[:,0]+ dx
        y = dataneg[:,1]+ dy
        t = dataneg[:,2]-1
        input[t ,1,x ,y ] = 1

        reshape = input[:,:,16:240,16:240]
        label = torch.tensor([classnum])
        return reshape, label.item()

    def __len__(self):
        return len(self.filenames)


class ImageNet_Fusion(data.Dataset):
    def __init__(self, train=True, fig=True, frame=True, data_dvs_path='../', data_fig_path='../', transform=None):
        super().__init__()
        
        self.fig = fig
        self.frame = frame
        self.data_dvs_path = data_dvs_path
        self.transform = transform
        
        self.train = train
        self.filenames = []
        self.trainpath = data_dvs_path+'train'
        self.testpath = data_dvs_path+'val'
        self.traininfotxt = data_dvs_path+'trainlabel_v2.txt'
        self.testinfotxt = data_dvs_path+'vallabel_v2.txt'
        self.formats = '.npz'
        if train:
            self.data_fig_path = data_fig_path + 'train/'
            self.path = self.trainpath
            trainfile = open(self.traininfotxt, 'r')
            for line in trainfile:
                filename, classnum, a, b, fig_path = line.split()
                realname,sub = filename.split('.')
                self.filenames.append(realname+self.formats)
        else:
            self.data_fig_path = data_fig_path + 'val/'
            self.path = self.testpath
            testfile = open(self.testinfotxt, 'r')
            for line in testfile:
                fig_path, filename, classnum, a, b = line.split()
                realname,sub = filename.split('.')
                self.filenames.append(realname+self.formats)

    def __getitem__(self, index):
        if self.train:
            info = linecache.getline(self.traininfotxt, index+1)
            filename, classnum, a, b, fig_path = info.split()
        else:
            info = linecache.getline(self.testinfotxt, index+1)
            fig_path, filename, classnum, a, b = info.split()
            fig_path = fig_path+'/'+filename
            fig_path = fig_path[:-4] + '.JPEG'
        
        realname,sub = filename.split('.')
        filename = realname+self.formats
        filename = self.path + r'/' + filename
        classnum = int(classnum)
        
        if self.frame:
            a = int(a)
            b = int(b)
            datapos = np.load(filename)['pos'].astype(np.float64)
            dataneg = np.load(filename)['neg'].astype(np.float64)

            dy = (254 - b) // 2
            dx = (254 - a) // 2
            input = torch.zeros([8, 2, 256, 256])

            x = datapos[:,0]+ dx
            y = datapos[:,1]+ dy
            t = datapos[:,2]-1
            input[t ,0,x ,y ] = 1

            x = dataneg[:,0]+ dx
            y = dataneg[:,1]+ dy
            t = dataneg[:,2]-1
            input[t ,1,x ,y ] = 1

            reshape = input[:,:,16:240,16:240]
        else:
            reshape = torch.ones([8, 2, 224, 224])
        
        
        if self.fig:
            img = default_loader(self.data_fig_path+fig_path)
            if self.transform is not None:
                img = self.transform(img)
        else:
            img = torch.ones([3, 224, 224])
        
        label = torch.tensor([classnum])

        return reshape, img, label.item()

    def __len__(self):
        return len(self.filenames)
    