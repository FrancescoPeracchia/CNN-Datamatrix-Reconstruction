
import torch
from matplotlib import image
from .cnn import EdgeDetector,Predictor
import os
from torchvision.utils import save_image
import math
import cv2
import torch.nn.functional as F
from PIL import Image
from pylibdmtx.pylibdmtx import decode
import torchvision

class Recostructor():

    def __init__(self,n_boxes,dim,cfg_eval = None) -> None:
        self.modes = ["train","test","inference"]
        self.M= n_boxes
        self.N = dim
        self.folder_dir_save = "/home/mv/Desktop/DM_reco/DM_reco/test"

        self.model_edg = EdgeDetector(self.N).cuda()
        self.model_rec = Predictor(self.N).cuda()

        if cfg_eval is not None :
            self.model_edg.load_state_dict(torch.load("model_predictor.pth"))
            self.model_rec.load_state_dict(torch.load("model_edge_dectector.pth"))
            self.model_edg.eval()
            self.model_rec.eval()
            print("loaded")

    def edge_gt(self,input : torch.tensor):

 

            Total_Boxes = self.M*self.M 


            input = torch.unsqueeze(input,0)
            x = torch.reshape(input,(1, self.M* self.M))
            list_horizontal = []
            #print(x)

            for box in range(Total_Boxes):

                if (box+1)%self.M == 0:
                    continue 
                
                left = torch.unsqueeze(x[0,box],0)
                right = torch.unsqueeze(x[0,box+1],0)
                if left == right :
                    new_c = 0
                else :
                    new_c = 1
                
                list_horizontal.append(new_c)
            


            list_vertical = []

            for box in range(Total_Boxes):

                if (box+self.M) >= Total_Boxes:
                    continue 


                upper = torch.unsqueeze(x[0,box],0)
                back = torch.unsqueeze(x[0,box+self.M],0)
                if upper == back :
                    new_v = 0
                else :
                    new_v = 1
                

                list_vertical.append(new_v)          


            list_horizontal = torch.tensor(list_horizontal, dtype=torch.long)
            list_vertical = torch.tensor(list_vertical, dtype=torch.long)
            gt_tensot_edge = torch.cat((list_horizontal,list_vertical), dim=0)

            return gt_tensot_edge
   
    def prepare_edge_input(self,input:torch.tensor):
            input_tensor = torch.unsqueeze(input,0)
            Total_Boxes = self.M* self.M

            unfo = torch.nn.Unfold((self.N,self.N), dilation=1, padding=0, stride=self.N)
            x = unfo(input_tensor)
            x = torch.reshape(x,(1,self.N,self.N, self.M* self.M))
            list_horizontal = []

            for box in range(Total_Boxes):
                img_path_save = str(os.path.join(self.folder_dir_save,"horizontal","imageh_"+str(box)+".png"))

                if (box+1)%self.M == 0:
                    continue 
                
                left = torch.unsqueeze(x[0,:,:,box],0)
                right = torch.unsqueeze(x[0,:,:,box+1],0)
                new_c = torch.unsqueeze(torch.cat((left, right), 2),0)
                       
                save_image(torch.reshape(new_c,(1,self.N,-1)),img_path_save)
                #print("new_v",new_c.size())
                list_horizontal.append(new_c)
            
            
            
            

 
            c = torch.cat(list_horizontal, dim=0)
            list_vertical = []

            for box in range(Total_Boxes):

                img_path_save = str(os.path.join(self.folder_dir_save,"vertical","image_"+str(box)+".png"))
                img_path_saveU = str(os.path.join(self.folder_dir_save,"verticalU","image_"+str(box)+".png"))
                img_path_saveB = str(os.path.join(self.folder_dir_save,"verticalB","image_"+str(box)+".png"))
                img_path_saveC = str(os.path.join(self.folder_dir_save,"verticalC","image_"+str(box)+".png"))
                img_path_saveV = str(os.path.join(self.folder_dir_save,"horizontalV","image_"+str(box)+".png"))


                if (box+self.M) >= Total_Boxes:
               
                    continue 

                
                upper = torch.unsqueeze(x[0,:,:,box],0)
                back = torch.unsqueeze(x[0,:,:,box+self.M],0)
                save_image(upper,img_path_saveU)
                save_image(back,img_path_saveB)
                cat_ = torch.cat((upper, back), 1)
                save_image(cat_,img_path_saveC)
                cat_ = torch.transpose(cat_,1,2)
                cat_ = torch.unsqueeze(cat_,0)
                save_image(cat_,img_path_saveV)
                
                save_image(cat_,img_path_save)
                list_vertical.append(cat_)

 
            v = torch.cat(list_vertical, dim=0)
 



            input_tensor_edge = torch.unsqueeze(torch.cat((c,v), dim=0),dim = 0)
            #print(input_tensor_edge.size())
            return input_tensor_edge
        
    def prepare_input_rec(self,input:torch.tensor):
            input_tensor = torch.unsqueeze(input,0)
            Total_Boxes = self.M* self.M
            NR = self.N*3

            #print("input_tensor",input_tensor.size())
            img_path_save = str(os.path.join(self.folder_dir_save,"patch","image_"".png"))
            img_path_save_avg = str(os.path.join(self.folder_dir_save,"patchAVG","image_"".png"))

            

            unfo = torch.nn.Unfold((NR,NR), dilation=1, padding=0, stride=self.N)
            x = unfo(input_tensor)
            x = torch.reshape(x,(1,NR,NR, -1))
            #print("unfold",x.size())
            testest = torch.unsqueeze(x,0)
            #print(testest.size())
            x = torch.transpose(testest,0,4)
            #print(x.size())
            x = torch.squeeze(x,-1)

            unfo0 = torch.nn.Unfold((self.N,self.N), dilation=1, padding=0, stride=self.N)

            x = unfo0(x)
            #print(x.size())
            x = torch.mean(x,dim =1) 
            #x = torch.unsqueeze(torch.unsqueeze(x[0,:],0),0)
            #print(x.size())
            x = torch.reshape(x,(-1,3,3))
            

                        
            
            #save_image(x,img_path_save_avg)
            #save_image(testest[:,:,:,:,0],img_path_save)

  







            return x

    def consistency(self,pathA:tuple,pathB:tuple,):
        #pathA
        patha = True
        if pathA[0] > 0.5:
            patha = not patha
        
        if pathA[1] > 0.5:
            patha = not patha
        

        #pathB
        pathb = True
        if pathB[0] > 0.5:
            pathb = not pathb
        
        if pathB[1] > 0.5:
            pathb = not pathb
        
        if patha == pathb:
            if patha == True:
                return 0
            else :
                return 1


        
        else :
            return None 

    def generate_image(self,image:torch.tensor,scale = 5):
        h,w = image.size()
        dimh = h*scale
        dimw = w*scale

        void = torch.ones((dimh+2,dimw+2),dtype = float)
        box = torch.zeros((scale,scale),dtype = float)
        
        for i in range(h):
            i_s = i*scale+1
            for j in range(w):
                j_s = j*scale+1
                if image[i,j] == 0:
                    void[i_s:i_s+scale,j_s:j_s+scale] = box

        return void

    def prepare_edge_rec(self,input:torch.tensor):
        
        lenght,_ = input.size()
        lenght  = int(lenght/2)
        lenghtDM  = int(math.sqrt(lenght))




        horizontal  = input[0:lenght,:]
        

        horizontal = torch.reshape(horizontal,(lenghtDM+1,lenghtDM,2))
        horizontal = horizontal[1:-1,:,:]
        #print(horizontal.size())


        vertical  = input[lenght:,:]
        vertical = torch.reshape(vertical,(lenghtDM,lenghtDM+1,2))
        vertical = vertical[:,1:-1,:]
        rows,_,_  = vertical.size()
        _,column,_  = horizontal.size()

        list_attention_edge  = []


        for i in range(rows-1):
            for j in range(column-1):

                verticalU = torch.unsqueeze(vertical[i,j,1],0)
                verticalB = torch.unsqueeze(vertical[i+1,j,1],0)
                horiL = torch.unsqueeze(horizontal[i,j,1],0)
                horiR = torch.unsqueeze(horizontal[i,j+1,1],0)




                

                flag_verticalPU = True
                flag_verticalPB = True
                flag_verticalSU = True
                flag_verticalSB = True
                flag_horiPR = True
                flag_horiPL = True
                flag_horiSR = True
                flag_horiSL = True


                #since we are processing from left to right from up to bottom
                try : verticalPU =  torch.unsqueeze(vertical[i,j-1,1],0)
                except : flag_verticalPU = False
                try : verticalPB =  torch.unsqueeze(vertical[i+1,j-1,1],0)
                except : flag_verticalPB = False
                try : verticalSU =  torch.unsqueeze(vertical[i,j+1,1],0)
                except : flag_verticalSU = False
                try : verticalSB =  torch.unsqueeze(vertical[i+1,j+1,1],0)
                except : flag_verticalSB = False



                try : horiPR = torch.unsqueeze(horizontal[i-1,j+1,1],0)
                except : flag_horiPR = False
                try : horiPL = torch.unsqueeze(horizontal[i-1,j,1],0)
                except : flag_horiPL = False
                try : horiSR = torch.unsqueeze(horizontal[i+1,j+1,1],0)
                except : flag_horiSR = False
                try : horiSL = torch.unsqueeze(horizontal[i+1,j,1],0)
                except : flag_horiSL = False



                #check conditions
                #UPPER LEFT
                list_diagonals = []
                dia = 0
                if flag_horiPL and flag_verticalPU :
                    #we can start searching
                    value = self.consistency((verticalU,horiPL),(horiL,verticalPU))
                    
                    if value == None:
                        dia = 0
                    else:
                        dia = value

                

                list_diagonals.append(dia)
                #UPPER RIGHT
                dia = 0
                if flag_horiPR and flag_verticalSU :
                    #we can start searching
                    value =  self.consistency((verticalU,horiPR),(horiR,verticalSU))
                    
                    if value == None:
                        dia = 0
                    else:
                        dia = value

                
                list_diagonals.append(dia)
                #BOTTOM LEFT
                dia = 0
                if flag_horiSL and flag_verticalPB :
                    #we can start searching
                    value =  self.consistency((verticalB,horiSL),(horiL,verticalPB))
                    
                    if value == None:
                        dia = 0
                    else:
                        dia = value
                

                list_diagonals.append(dia)
                #BOTTOM RIGHT
                dia = 0
                if flag_horiSR and flag_verticalSB :
                    #we can start searching
                    value =  self.consistency((verticalB,horiSR),(horiR,verticalSB))
                    
                    if value == None:
                        dia = 0
                    else:
                        dia = value
                

                
                list_diagonals.append(dia)

                #print(list_diagonals)
                    



       

            

                #first_row = torch.cat((0, verticalU, 0), 0)
                #print(first_row)
                #torch.cat((torch.zeros(1,dtype = float).cuda(),verticalU,torch.zeros(1,dtype = float).cuda()),0)
                """
                rawf = torch.unsqueeze(torch.cat((torch.zeros(1,dtype = float).cuda(),verticalU,torch.zeros(1,dtype = float).cuda()),0),0)
                raws =  torch.unsqueeze(torch.cat((horiL,torch.zeros(1,dtype = float).cuda(),horiR),0),0)
                rawt = torch.unsqueeze(torch.cat((torch.zeros(1,dtype = float).cuda(),verticalB,torch.zeros(1,dtype = float).cuda()),0),0)

                """
                rawf = torch.unsqueeze(
                    torch.cat(
                    (torch.unsqueeze(torch.tensor(list_diagonals[0],dtype = float).cuda(),0),
                    verticalU,
                    torch.unsqueeze(torch.tensor(list_diagonals[1],dtype = float).cuda(),0))    
                    ,0)
                    ,0)


                raws =  torch.unsqueeze(
                    torch.cat(
                        (horiL,
                        torch.zeros(1,dtype = float).cuda(),
                        horiR)
                        ,0)
                        ,0)
                
                rawt = torch.unsqueeze(
                    torch.cat(
                    (torch.unsqueeze(torch.tensor(list_diagonals[2],dtype = float).cuda(),0),
                    verticalB,
                    torch.unsqueeze(torch.tensor(list_diagonals[3],dtype = float).cuda(),0))
                    ,0)
                    ,0)
                

                #print(rawf.size())
                #print(raws.size())
                #print(rawt.size())

                attention_edge  = torch.unsqueeze(torch.cat((rawf,raws,rawt),0),0)



                list_attention_edge.append(attention_edge)
                #print(attention_edge)
                
            

        return  torch.cat(list_attention_edge, 0)
        
    def build(self,input : dict, mode :str):
        assert mode in self.modes 


       

        if mode == "train":

            
       
            #print(input["img_path"])
            input_edg = self.prepare_edge_input(input["image"][0])
            gt_edg = self.edge_gt(input["gt"][0]).cuda()

            output,loss_edge_detection = self.model_edg(input_edg,gt_edg)

            gt_image = input["gt"][0]
            rec_img = self.prepare_input_rec(input["image"][0])
            rec_edg = self.prepare_edge_rec(output)
            #print("rec image",rec_img)
            mapped = self.model_rec(rec_img,rec_edg,gt_image)


            gt_imageGT = gt_image.detach().clone()


            gt_image[1:-1,1:-1] = mapped

          
            
            #tensor  = torch.unsqueeze(gt_image,0).cpu().numpy() # make sure tensor is on cpu
            #cv2.imwrite(tensor,input["img_path_save_pre"])
            
            save_image(torch.unsqueeze(gt_image,0),input["img_path_save_pre"][0])
            #print(gt_image.size())
            padded= self.generate_image(gt_image)

        
            save_image(torch.unsqueeze(padded,0),input["img_path_save_pre_pad"][0])
            #print(input["img_path_save_pre_pad"][0])
            #print(decode(Image.open(input["img_path_save_pre_pad"][0])))

            


            gt_image = torch.flatten(gt_image)
            gt_imageGT = torch.flatten(gt_imageGT)

            

            correct = (gt_image == gt_imageGT).float().sum()
        
            print("accuracy with BOARDERS:",correct/gt_imageGT.size()[0])


           




        
            return output,loss_edge_detection
        

        elif  mode == "inference":
            self.M = input["shape"][0]
            c,h,w = input["image"].size()
            self.N = int(h/self.M)
            print(self.N)

            input_edg = self.prepare_edge_input(input["image"])
            output = self.model_edg(input_edg,None)
            gt_image = input["gt"]
            rec_img = self.prepare_input_rec(input["image"])
            rec_edg = self.prepare_edge_rec(output)
            print("GT",gt_image.size())
            mapped = self.model_rec(rec_img,rec_edg,gt_image)
            gt_image[1:-1,1:-1] = mapped
            padded= self.generate_image(gt_image)
            save_image(torch.unsqueeze(padded,0),input["path_output"])
            input["path_output"]




        elif mode == "test":
            pass




            

        
        

