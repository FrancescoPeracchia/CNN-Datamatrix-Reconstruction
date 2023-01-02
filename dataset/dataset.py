
from matplotlib.ft2font import HORIZONTAL
from torchvision import transforms
import os
import pandas as pd
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt
import json
from PIL import Image
from torchvision.utils import save_image

class DMdataset(Dataset):

    def __init__(self,cfg_dataset,mode):
    
        self.folder_dir = cfg_dataset.folder_dir

        self.folder_dir_save = cfg_dataset.folder_dir_save
        self.folder_dir_save_pre = cfg_dataset.folder_dir_save_pre
        self.img_path_save_pre_pad = cfg_dataset.img_path_save_pre_pad
        print(cfg_dataset.folder_dir_save)

        self.transform_cfg = cfg_dataset.transform

        self.newsize = cfg_dataset.transform.size_resize_img

        annotation_file = open(os.path.join(self.folder_dir,"annotationsGT.json"))
        self.annotation = json.load(annotation_file)

        ColorJitter_dict = self.transform_cfg.ColorJitter
        RandomPerspective_dict = self.transform_cfg.RandomPerspective
        GaussianBlur_dict = self.transform_cfg.GaussianBlur
        resize_dim = self.transform_cfg.size_resize_img
        self.dim_boxes = 12

       
        self.tranformation_list = [transforms.GaussianBlur(GaussianBlur_dict.kernel_size,GaussianBlur_dict.sigma),
                            transforms.Resize(resize_dim),
                            transforms.ColorJitter(brightness=ColorJitter_dict.brightness,contrast=ColorJitter_dict.contrast, saturation=ColorJitter_dict.saturation, hue=ColorJitter_dict.hue),
                            transforms.Grayscale(num_output_channels = 1),
                            transforms.ToTensor()]
   
        self.transform = transforms.Compose(self.tranformation_list)
        self.mode = mode
        assert mode in ["train","val","test"]

    def __len__(self):
        return len(self.annotation[self.mode])

    def __getitem__(self, idx):



        data = {}
        """
        525 images and 535 annotations(barcodes), there are some images with mulitple barcode
        use idx to read the associate image id and all the annotations attached.
        """


        #Image reading and transformation if needed
        img_path = os.path.join(self.folder_dir, self.annotation[self.mode][idx]["hd"])
        img_path_save = str(os.path.join(self.folder_dir_save,self.annotation[self.mode][idx]["id"]))+".png"
        img_path_save_pre = str(os.path.join(self.folder_dir_save_pre,self.annotation[self.mode][idx]["id"]))+".png"
        img_path_save_pre_pad = str(os.path.join(self.img_path_save_pre_pad,self.annotation[self.mode][idx]["id"]))+".png"

        image = Image.open(img_path)

        
        w,h = image.size
        image_original_shape = (h,w)

        data["img_path"] = img_path
        data["image_original_shape"] = image_original_shape
        data["img_path_save_pre"] = img_path_save_pre
        data["img_path_save_pre_pad"] =  img_path_save_pre_pad                    


        image_id = self.annotation[self.mode][idx]["id"]
        gt = self.annotation[self.mode][idx]["gt"]

         


        image = self.transform(image)
        


        save_image(image,img_path_save)



      

        data["image"] = image.cuda()
        data["gt"]  = torch.FloatTensor(gt)

       
        return data
    

    def prepare_image(self, image_path,shape):

        data = {}
        new_size = (int(shape[1]*self.dim_boxes),int(shape[1]*self.dim_boxes))
        self.tranformation_list 
        tranformation_list = [transforms.GaussianBlur((3,3),1),
                    transforms.Resize(new_size),
                    transforms.Grayscale(num_output_channels = 1),
                    transforms.ToTensor()]
        transform = transforms.Compose(tranformation_list)
        image = Image.open(image_path)
        image = transform(image)
        data["image"] = image.cuda()
        #gt = self.annotation[self.mode][0]["gt"]
        gt = torch.zeros([shape[0], shape[1]])
        
        #Horizontal
        flag_black = True
        list_hor = []
        for i in range(shape[0]):
            if flag_black :
                list_hor.append(0)
                flag_black = False

            else :
                list_hor.append(1)
                flag_black = True
        
        #Vertical
        flag_black = False
        list_vertical = []
        for i in range(shape[0]):
            if flag_black :
                list_vertical.append(0)
                flag_black = False

            else :
                list_vertical.append(1)
                flag_black = True
        
       
        verti = torch.tensor(list_vertical, dtype=torch.long)
        hor = torch.tensor(list_hor, dtype=torch.long)

        gt[0,0:] = hor
        gt[0:,-1] = verti





        

        data["gt"]  = torch.FloatTensor(gt)
        

        return data 








 



    