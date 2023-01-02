import imp
from unicodedata import name
import os
from cv2 import accumulate
from torch.optim.lr_scheduler import ExponentialLR
from torchvision.utils import save_image
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToPILImage
import numpy
import cv2
"""
def init_dictionary(dictionary):
    for class_ in classes :
        start_ = {"score":0,
                "boxes":[],
        }
        dict[str(class_)] = start_
    return dictionary
"""


def weighing_loss(losses_dictionary,loss_weights):
    loss = 0
    #print(len(losses_dictionary))
    #print(len(loss_weights))
    assert len(losses_dictionary) == len(loss_weights)
    for j,loss_type in enumerate(losses_dictionary):
        loss += losses_dictionary[loss_type]*loss_weights[j]
    
    losses_dictionary["loss_total"] = loss

    return losses_dictionary

def process_results_batch(epochs_information,new_info_to_integrate = None,batch_len = None):

    if new_info_to_integrate != None:
        flag_init = False

        #check if is the first batch of the epoch, in this case we should initialize the dict
        if len(epochs_information) == 0:
            flag_init = True



        for key in new_info_to_integrate:
            new = new_info_to_integrate[key]
            if  flag_init :
                epochs_information[key] = new
            else:
                epochs_information[key] += new

        return epochs_information
    else:
        assert batch_len != None
        
        for key in epochs_information:
            epochs_information[key] = epochs_information[key]/batch_len
        
        return epochs_information

def filter_output(labels,scores,boxes):
    filtered = dict()
    #filtered = init_dictionary(filtered)

    for i,label in enumerate(labels) : 
        current_label = label

        try : filtered[str(current_label)]
        except : 
            filtered[str(current_label)] = {"score":0,"boxes":[]}


        if scores[i] > filtered[str(current_label)]["score"]:
            filtered[str(current_label)]["score"] = scores[i]
            filtered[str(current_label)]["boxes"] = boxes[i]
    
    return filtered



def Tloop(cfg_training,model,dataloaders):
    """
    cfg : configuration file
    model : pytorch model
    dataloaders : list of train. validation datasets [train,validation]
    """
    writer = SummaryWriter()


    #send model to GPUs
    counter = 0
    optimizer = torch.optim.SGD(model.model_edg.parameters(), lr=0.001)
    scheduler = ExponentialLR(optimizer, gamma=0.94)

    model.model_edg.train()
    EPOCH = 20

    for epochs in range(EPOCH):
        model.model_edg.train()
        print("Epoch : ",str(epochs)+"/"+str(EPOCH))
        dict_epochs_information = {}
        running_loss = 0
        counter = 0
        accumulatation_step = 0
            
        for data in dataloaders:


            """
            training on train set
            """
            
            optimizer.zero_grad()
           

            x,lossEdge = model.build(data,"train")
            losses  =  lossEdge 
            #print("losses : ",losses)
            #print("lossEdge : ",lossEdge)
          
            losses.backward()
            
            optimizer.step()





        scheduler.step()
    

        PATH0 = "model_predictor.pth"
        PATH1 = "model_edge_dectector.pth"
        torch.save(model.model_rec.state_dict(),PATH1)
        torch.save(model.model_edg.state_dict(),PATH0)
        