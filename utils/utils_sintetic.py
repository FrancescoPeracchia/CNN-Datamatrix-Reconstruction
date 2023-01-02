import numpy as np
import cv2
import numpy as np
from pathlib import Path
import json
import datetime
from typing import List

def  draw_infos_image(image_GT : np.array,list_gt: list,color = (255,255,0),thickness = 1):

    image = np.array(image_GT, copy=True)  

    #print(list_gt)
    list_ = []
    counter = 0
    for object in list_gt :

        list_char_anno = object["Annotations"]
        list_offset = object["IMG_OFF"] 
        offsets = len(list_offset)
        
        bbox = object["BBOX"]
        A = bbox[0] 
        B = bbox[1]
        color = (0,255,255)
        image = cv2.rectangle(image,A,B,color,4)
        #image = cv2.putText(image, "BBOX", A+20 , cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

  
        bbox = object["BBOX_FULL"]
        A = bbox[0] 
        B = bbox[1]
        color = (255,255,0)
        image = cv2.rectangle(image,A,B,color,2)
        #image = cv2.putText(image, "BBOX_FULL", A , cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        color = (255,0,0)


        if len(list_char_anno) > 0: 
            for item in list_char_anno:
                bbox = item["bbox"]
                char = item["char"]

                A = bbox[0] 
                B = bbox[1]
  


                for o in range(offsets):

                

                    A = (A[0]+list_offset[o][0],A[1]+list_offset[o][1])
                    B =  (B[0]+list_offset[o][0],B[1]+list_offset[o][1])
                    
                image = cv2.rectangle(image,A,B,color,thickness)
                #image = cv2.putText(image, char, A , cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                #print("CHAR",char)
            
                counter += 1

                box = {"points":(A,B),"counter":counter}
                list_.append(box)

                #image = cv2.circle(image, A, radius=0, color=(0, 255, 255), thickness=-1)
                #image = cv2.circle(image, B, radius=0, color=(255, 0, 255), thickness=-1)
            

        #bbox over image
        

        else:
            pass
    

    #print("list",list_)
        
        

        
    return image

def pre_merge_resize(image1_dim, image2_dim, image2, gt):
    """
    used before merging horizontally. to match y dimentions
    """
    y_targer = image1_dim[0]
    y_new = image2_dim[0]

    scale_percent = (y_targer / y_new) * 100

    width = int(image2_dim[1] * scale_percent / 100)
    height = int(y_new * scale_percent / 100)
    # to avoid fractions we force height to be exacly the targert, otherwise could 
    dim = (width, height)
    #print("target ", image1_dim)
    #print("new ", image2_dim)

    #print("pre ", image2.shape)
    #print("dim ", dim)
    resized = cv2.resize(image2, dim, interpolation = cv2.INTER_AREA)
    #print("resized", resized.shape)
    
    #gt rescale
    #gt is a list of dictionary with "char" and "bbox"
    new_list_gt = []
    for gt_element in gt :

        bbox = gt_element["bbox"]

        #up_left = bbox[0]
        #bottom_right = bbox[1]

        #same scaling along x,y axis
 
        bbox_ =[]
        for point in bbox:
            new_x_c = int(point[0] * scale_percent / 100)
            new_y_c = int(point[1] * scale_percent / 100)

            
            bbox_.append((new_x_c, new_y_c))

        
        element = {"char": gt_element["char"], "bbox": bbox_}
        new_list_gt.append(element)
    
    return resized, new_list_gt

class Padder():
    def __init__(self, list: list):

        list_string = ""
        for l in list :
            list_string += str(l)
        
        self.init_list = list
        self.list_string = list_string

    def padd(self,dim_paddind, fillvalue=0):
        list_pad = [fillvalue] * (dim_paddind - len(self.init_list)) + self.init_list, 
        p = ""
        for i in range((dim_paddind - len(self.init_list))):
            p += str(fillvalue)
        

        list_str_pad = p + self.list_string
        return list_pad,list_str_pad
    
    def add_space_text(self, text):
        processed = " "
        for t in text:
            processed += t + " "
        
        #return processed[1:-1]
        return processed



############################################################

# https://cocodataset.org/#format-data
# https://opencv.org/introduction-to-the-coco-dataset/
class Annotator():
    def __init__(self, 
                    year: int=datetime.date.today().year, 
                    version: str="",
                    description: str="",
                    contributor: str="",
                    url: str="",
                    date_created: datetime=datetime.date.today() ):

        self.info = {
            "year": datetime.date.today().year,
            "version": version,
            "description": description,
            "contributor": contributor,
            "url": url,
            "date_created": date_created,
        }
        self.images = []
        self.annotations = []
        self.categories = []
        self.licenses = []


    def add_license(self,
                    url: str="",
                    id: int=0,
                    name: str="") -> None:
        license = {
            "url": url,
            "id": id,
            "name": name,
        }
        self.licenses.append(license)
    

    def add_image(self, 
                    id: int, 
                    width: int, 
                    height: int, 
                    file_name: str, 
                    license: int=0, 
                    date_captured: datetime=datetime.datetime.now() ) -> None:
        image = {
            "id": id, 
            "width": width,
            "height": height,
            "file_name": file_name,
            "license": license,
            "date_captured": date_captured,
        }
        self.images.append(image)

        
    def add_annotations(self, 
                    id: int, 
                    image_id: int, 
                    category_id: int, 
                    bbox: List[int] ) -> None: # xmin, ymin, width, height 
        annotation = {
            "id": id, 
            "image_id": image_id,
            "category_id": category_id,
            "bbox": bbox,
        }
        self.annotations.append(annotation)



    def add_category(self, 
                    id: int, 
                    name: str, 
                    supercategory: str ) -> None:
        category = {
            "id": id, 
            "name": name,
            "supercategory": supercategory
            }
        self.category.append(category)

