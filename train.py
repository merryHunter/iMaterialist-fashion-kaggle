import torch
import torch.optim as optim
import json
import numpy as np
import cv2
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
import squeezenet as sq
import os
from PIL import Image
from copy import deepcopy
import torchvision
import math

n_classes = 229 # n_classes + 1 (one class for all zeros in one hot encoding)
BATCH_SIZE = 8

class FashionDataset(Dataset):
    def __init__(self, json_path, img_path, set_type="train"):
        """
        Args:
            json_path (string): path to json file
            img_path (string): path to the folder where images are
            set_type: train or test , val is not supported yet
        """
        self.set_type = set_type
        self.to_tensor = transforms.ToTensor()
        self.data_info = json.loads(open(json_path).read())
        self.label_count = {x: 0 for x in range(229)}
        
        # read jsons
        arrr = [] # labels in one hot encoding,
        imgs = []
        names = []
        i = 0
        if set_type != "test":
            for ann in self.data_info['annotations']:
#                if i < 10000:
#                    continue
                imgs.append(os.path.join(img_path, ann['imageId'] + '.jpg'))
                names.append(ann['imageId'])
                label_arr = []
                label_arr =  np.sum([np.eye(n_classes, dtype="uint8")
                                     [ int(x)] for x in ann['labelId'] ], axis=0, dtype="uint8").tolist() 
                arrr.append(deepcopy(label_arr))
                for x in ann['labelId']:
                        self.label_count[int(x)] +=1
                i +=1
                if i % 100000 == 0:
                    print("Processed: " + str(i))
#                    break
        else:
            for ann in self.data_info['images']:
                imgs.append(os.path.join(img_path,str(ann['imageId']) + '.jpg'))
                names.append(ann['imageId'])
                
        self.names = names
        self.image_arr = imgs
        # Second column is the labels
        
#         self.label_arr = torch.stack(arrr) 
        if set_type != "test":
            self.label_arr = arrr 
        
        # calculate class weight
#        print(self.label_count)
        if set_type == "train":
            self.class_weight = self._get_class_weight()
            print("Class weights:{0}".format(self.class_weight))
        # Calculate len
        self.data_len = len(self.image_arr)
        self.transformations = transforms.Compose([transforms.Pad(8),transforms.CenterCrop(224),
                                transforms.ToTensor()])
            
    def _get_class_weight(self):
        mu = 2
        total = sum(self.label_count.values())
        keys = self.label_count.keys()
        local_class_weight = dict()
         
        for key in keys:
            score = math.log(mu*total/float(self.label_count[key]+1))
            local_class_weight[key] = score if score > 1.0 else 1.0

        return local_class_weight
    
    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open image
        img_as_img = Image.open(single_image_name)

        # Check if there is an operation
#         some_operation = self.operation_arr[index]
        # Transform image to tensor
        
#         img_as_tensor = self.to_tensor(img_as_img)
        img_as_tensor = self.transformations(img_as_img)  #
        
        
        if self.set_type == "test":
            return (img_as_tensor, self.names[index])
        
        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]
        return (img_as_tensor, torch.cuda.FloatTensor(single_image_label), self.names)

    def __len__(self):
        return self.data_len



#============================
#======== data loader =======

my_imgs =  FashionDataset(json_path='train.json',img_path="/unreliable/DATASETS/chernuka/fashion/train", set_type="train")
mn_dataset_loader = torch.utils.data.DataLoader(dataset=my_imgs,
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=False)
my_test_imgs =  FashionDataset(json_path='test.json',img_path="test/", set_type="test")
test_dataset_loader = torch.utils.data.DataLoader(dataset=my_test_imgs,
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=False)


model = sq.squeezenet1_1(num_classes=n_classes)
#model = torch.nn.DataParallel(model).cuda()
#model.load_state_dict(torch.load('models/squeezenet-bceloss-10ks.pt'))
#print('loading model')
"""
pretrained_dict = torch.load('models/mytraining-squeezenet-finetuned.pt') #squeezenet1_1-f364aa15.pth')
model_dict = model.state_dict()

# 1. filter out unnecessary keys
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# remove last layer, as we have 229 labels, not 1000 as in ImageNet
pretrained_dict.pop('classifier.1.weight', None)
pretrained_dict.pop('classifier.1.bias', None)

# 2. overwrite entries in the existing state dict
#model_dict.update(pretrained_dict) 
# 3. load the new state dict
#model.load_state_dict(model_dict)

j = 0
for param in model.parameters():
    param.requires_grad = False
    j+=1
    if j >= 50: # TODO
        param.requires_grad = True
"""
# criterion = torch.nn.BCELoss(weight=my_imgs.class_weight) #MultiLabelSoftMarginLoss()
# filter out those without grad
# optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.00001, momentum=0.9)


epochs = 1
def run(epochs=1, i=0):
    for epoch in range(epochs):
        losses = []
        for images, labels, _ in mn_dataset_loader:
            inputv = Variable(images)
#            print(labels)
#            labelsv = Variable(torch.cuda.FloatTensor(labels))
            labelsv = Variable(labels)
            output = model(inputv)
            loss = criterion(output, labelsv)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.data.mean())
            if i % 1000 == 0:
                print('[%d/%d] Loss: %.3f' % (epochs, i, loss.data.mean()))
                torch.save(model.state_dict(), 'models/inception-multilabel-all-%d.pt' % i )
#                 break
            i += 1
        print('[%d/%d] Loss: %.3f' % (epoch+1, epochs, np.mean(losses)))

#run(2)
#for param in model.parameters():
#    param.requires_grad = True


#model_conv = torchvision.models.resnet18(pretrained=True)
model_conv = torchvision.models.inception_v3(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = torch.nn.Linear(num_ftrs, n_classes)
model = model_conv
model = torch.nn.DataParallel(model).cuda()

# criterion = torch.nn.MultiLabelSoftMarginLoss(weight=self.class_weight)
ww = np.asarray([list(my_imgs.class_weight.values()) for i in range(BATCH_SIZE)])
#w = torch.from_numpy(ww).cuda()
w = torch.cuda.FloatTensor(ww)
criterion = torch.nn.MultiLabelSoftMarginLoss(weight=w)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0005)


run(1, 2000)
#model.load_state_dict(torch.load('models/inception-multilabel-14000.pt'))
torch.save(model.state_dict(), 'models/inception-multilabel-all-weighted.pt')



label_ids = []
i = 0
for images, names in test_dataset_loader:
    inputv = Variable(images)
    outputs = model(inputv)
    pred = torch.sigmoid(outputs).data > 0.1 # define threshold
    if len(pred) == BATCH_SIZE:
        label_ids.extend(np.split(pred, BATCH_SIZE))
    else:
        rest = len(pred)
        label_ids.extend(np.split(pred, rest))
    i += 1
    if i % 100 == 0:
        print(i)

preds = pd.read_csv("sample_submission.csv")


pred_ids = []
for label in label_ids:
    l = np.nonzero(label.numpy()[0])[0].tolist()
    o = ' '.join(str(x) for x in l)
    pred_ids.append(o)

preds['label_id']= pred_ids
preds.to_csv('submissions/inception-multilabel-all-weigthed10.csv', index=False, header=True)
