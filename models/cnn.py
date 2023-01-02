
from numpy import kaiser, reshape
import torch.nn as nn 
import torch


class EdgeDetector(nn.Module):

    def __init__(self,dim_boxes) -> None:
        super().__init__()

        #stride = 1
        #dilation = 1
        #no padding
        #H = Hin - 2*padding - (k-1)
        #H = Hin - (k-1)
        #H = Hin+1-k

        #self.conv0 = nn.Conv2d(in_channel = 1,out_channel = 3, kernel_size = (1,5))
        self.N = dim_boxes

        channel = 6
        self.b_init = nn.BatchNorm2d(1)
        self.conv0 = nn.Conv2d(1,3,(5,5))
        self.b0 = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(3,channel,(3,3))
        self.b1 = nn.BatchNorm2d(6)

        #self.conv2 = nn.Conv2d(6,12,(3,3))
        #self.b2 = nn.BatchNorm2d(12)
        self.relu =  nn.ReLU()


    

        F =  int((self.N-(5-1)-(3-1))*(self.N-(5-1)-(3-1))*channel)*2
     
        F0 = int(F/2)
        

        self.FC0 = nn.Linear(F, F0)
        self.dropuout = nn.Dropout(p=0.3)
        self.FC1 = nn.Linear(F0, 2)



        self.criterion = torch.nn.NLLLoss()
        self.logSoft = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x,gt:None):

        #torch.Size([760, 4, 8])
        #torch.Size([760])
        batch,windows,channels,_,__ = x.size()
        if self.training:
            x = torch.reshape(x,(760*2,self.N,self.N))
            #print(x.size())
            x = torch.unsqueeze(x,dim =1)
            #print(x.size())
            x = self.b_init(x)
            x = self.conv0(x)
            x = self.b0(x)
            #print(x.size())
            x = self.relu(x)
            
            
            x = self.conv1(x)
            #print(x.size())
            x = self.b1(x)
            x = self.relu(x)
        

            x = torch.reshape(x,(760,-1))
            x = self.dropuout(x)
            x = self.FC0(x)
            #print(x.size())
            x = self.dropuout(x)
            x = self.FC1(x)
            #print(x.size())

            x_ = self.softmax(x)
            #print(x[1])
            pre = torch.argmax(x_,dim = 1)
            #print(gt.size())
            #print(pre[1])

            x = self.logSoft(x_)
            #print(gt.size())


            loss = self.criterion(x,gt)


            correct = (pre == gt).float().sum()
            print("accuracy:",correct/gt.size()[0])
            

            
        
            



            return x_,loss
        
        else :
            
            x = torch.reshape(x,(windows*2,self.N,self.N))
            #print(x.size())
            x = torch.unsqueeze(x,dim =1)
            #print("before",x)
            #print("before",x.size())
            x = self.b_init(x)
            #print("after",x)
            x = self.conv0(x)
            x = self.b0(x)
            #print(x.size())
            x = self.relu(x)
            
            
            x = self.conv1(x)
            #print(x.size())
            x = self.b1(x)
            x = self.relu(x)
        

            x = torch.reshape(x,(windows,-1))
            x = self.dropuout(x)
            x = self.FC0(x)
            #print(x.size())
            x = self.dropuout(x)
            x = self.FC1(x)
            #print(x.size())

            x_ = self.softmax(x)
            #print(x[1])
            pre = torch.argmax(x_,dim = 1)
            #print(gt.size())
            #print(pre[1])

            x = self.logSoft(x_)
            #print(gt.size())



            return x_
       
class Predictor(nn.Module):

    def __init__(self,dim_boxes) -> None:
        super().__init__()


        self.FC1 = nn.Linear(9, 2).cuda()
        self.binary_score = nn.Linear(1, 1).cuda()


        self.criterion = torch.nn.NLLLoss()
        self.logSoft = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.regression = nn.MSELoss()
    
    def computeAVG(self,values,list_output):
        list_black = []
        list_white = []
        
        for i,value in enumerate(values):
            if list_output[i] == 0:
                list_black.append(torch.unsqueeze(value,0))
            
            else:
                list_white.append(torch.unsqueeze(value,0))
        
        #print("list_white",list_white)
        list_white  = torch.cat(list_white,0)
        list_black  = torch.cat(list_black,0)
        
        return torch.mean(list_black, 0),torch.mean(list_white, 0)

    def correction(self,image,list,avgB,avgW):
 
        list_correnction = []


        
        for i,pre in enumerate(list):


            if pre != 2 :
                list_correnction.append(torch.unsqueeze(pre,0))
            
                
            else:
                values = image[i]
                differece_b = torch.abs(values - avgB)
                differece_w = torch.abs(values - avgW)
            
                if differece_b < differece_w:
                    list_correnction.append(torch.zeros(1,dtype = float))
                else:
                    list_correnction.append(torch.ones(1,dtype = float))
    
        
        
        list_correnction  = torch.cat(list_correnction,0)
        list_correnction = torch.squeeze(list_correnction,dim =-1)


        return list_correnction

    def forward(self,image,edges_prob,gt):
        
        #print(image.size())
        #print(edges_prob.size())
        #print(edges_prob[0])
        list_output = []
        counter = 0

        if self.training :

            for i,img in enumerate(image):
                img = torch.flatten(img)
                edges = torch.flatten(edges_prob[i])
                
                list_acc = []
                list_disacc = []
                ma = None
                md = None
                

                for j, edg in  enumerate(edges):
                    #print(edg) 
                    if edg == 0:
                        #print("skip")
                        pass
                    else :
                        #process edge probability
                        if edg > 0.5:
                            list_disacc.append(torch.unsqueeze(img[j],0))
                        #no edge
                        else :
                            list_acc.append(torch.unsqueeze(img[j],0))
                if len(list_disacc)== 0:
                    pass
                else:
                    list_disacc  = torch.cat(list_disacc,0)
                    md = torch.mean(list_disacc, 0)
                    
                
                if len(list_acc)== 0:
                    pass
                else:
                    list_acc  = torch.cat(list_acc,0)
                    ma = torch.mean(list_acc, 0)
                
            
                
                    
                if (ma != None and md != None): 
                    if (md > ma):
                        #disaccordg bboxes are more white
                        #is black
                        list_output.append(torch.unsqueeze(torch.zeros(1,dtype = float),0))
                    else  :
                        #disaccordg bboxes are more black
                        #is white
                        list_output.append(torch.unsqueeze(torch.ones(1,dtype = float),0))
                
                else:
                    counter += 1
                    unknown = torch.tensor([2], dtype=torch.float)
                    list_output.append(torch.unsqueeze(unknown,0))
                
                #print(edges)
                #print(img)
                #print(ma)
                #print(md)
                
                
                
                
                
            
            
            #print("NOT available n.",counter)
                        
            list_output  = torch.cat(list_output,0)
            list_output = torch.squeeze(list_output,dim =-1)
            #print("FINAL",list_output.size())


            values = image[:,1,1]
            avgB,avgW = self.computeAVG(values,list_output)
            copy = list_output.detach().clone()
            list_correnction = self.correction(values,copy,avgB,avgW)
            #print("avgB",avgB)
            #print("avgW",avgW)
            
            
            
            #correct by means 
            gt = gt[1:-1,1:-1]
            H,W = gt.size()
            gt = torch.flatten(gt)
            correct = (list_correnction == gt).float().sum()
            


            print("accuracy BOXES:",correct/gt.size()[0])

            mapped =  torch.reshape(list_correnction,(H,W))    
            
            return mapped

        else :

            for i,img in enumerate(image):
                img = torch.flatten(img)
                edges = torch.flatten(edges_prob[i])
                
                list_acc = []
                list_disacc = []
                ma = None
                md = None
                

                for j, edg in  enumerate(edges):
                    #print(edg) 
                    if edg == 0:
                        #print("skip")
                        pass
                    else :
                        #process edge probability
                        if edg > 0.5:
                            list_disacc.append(torch.unsqueeze(img[j],0))
                        #no edge
                        else :
                            list_acc.append(torch.unsqueeze(img[j],0))
                if len(list_disacc)== 0:
                    pass
                else:
                    list_disacc  = torch.cat(list_disacc,0)
                    md = torch.mean(list_disacc, 0)
                    
                
                if len(list_acc)== 0:
                    pass
                else:
                    list_acc  = torch.cat(list_acc,0)
                    ma = torch.mean(list_acc, 0)
                #print(edges)
                #print(img)
                #print(ma)
                #print(md)
                
            
                
                    
                if (ma != None and md != None): 
                    if (md > ma):
                        #disaccordg bboxes are more white
                        #is black
                        list_output.append(torch.unsqueeze(torch.zeros(1,dtype = float),0))
                    else  :
                        #disaccordg bboxes are more black
                        #is white
                        list_output.append(torch.unsqueeze(torch.ones(1,dtype = float),0))
                
                else:
                    counter += 1
                    unknown = torch.tensor([2], dtype=torch.float)
                    list_output.append(torch.unsqueeze(unknown,0))
                
                
                
                
            
            
            #print("NOT available n.",counter)
                        
            list_output  = torch.cat(list_output,0)
            list_output = torch.squeeze(list_output,dim =-1)
            #print("FINAL",list_output.size())


            values = image[:,1,1]
            print(values)
            print(list_output)
            avgB,avgW = self.computeAVG(values,list_output)
            copy = list_output.detach().clone()
            list_correnction = self.correction(values,copy,avgB,avgW)
            #print("avgB",avgB)
            #print("avgW",avgW)
            gt = gt[1:-1,1:-1]
            H,W = gt.size()
            

            mapped =  torch.reshape(list_correnction,(H,W))    
            
            return mapped
       


















        

