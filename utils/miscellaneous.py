import time
from tkinter.messagebox import NO
from tokenize import String
from abc import ABCMeta, abstractmethod
import cv2
import numpy as np


class HZ:
    def __init__(self, average_window=5) -> None:
        self.avg_len = average_window
        self.measurement = []
        self.starting_time = None
        self.ending_time = None
    
    def start(self):
        self.starting_time = time.time()
    

    def end(self):
        assert self.starting_time != None, print("call before HZ.start()")
        self.ending_time = time.time()
        hz = (1/(self.ending_time - self.starting_time))
        return self.frame_rate(hz)

    def frame_rate(self, measurement):
        self.measurement.insert(0, measurement)
        if len(self.measurement) + 1 == self.avg_len:
            self.measurement.pop()
        
        summed = sum(self.measurement)
        return summed/len(self.measurement)



class Cropper_Base():
    def __init__(self, mode) -> None:
        self.list_models = ["center", "AB"]
        assert mode in self.list_models, print("ERROR  only accepted models are :", self.list_models)
        self.mode = mode

    @abstractmethod
    def crop_by_center(self, image, **kargs):
        pass
        
    @abstractmethod        
    def crop_by_poits(self, image, **kargs):
        pass

    @abstractmethod
    def crop_by_center_in_new_scale(self ,image, **kargs):
        pass
        
    @abstractmethod        
    def crop_by_poits_in_new_scale(self, image, **kargs):
        pass
        


    def crop(self, image, change_mode = None, **kargs):
        #print("kargs",kargs)
        if change_mode != None:
            assert change_mode in self.list_models, print("ERROR  only accepted models are :", self.list_models)
            self.mode = change_mode

        if self.mode == "center":
            try: 
                return self.crop_by_center(image, **kargs)
            except:
                print("ERROR crop_by_center")
                
        elif self.mode == "AB":
            try:             
                return self.crop_by_poits(image, **kargs)
            except:
                print("ERROR crop_by_poits")  
    
 

class Cropper(Cropper_Base):

    #
    def __init__(self, mode: str) -> None:
        super(Cropper, self).__init__(mode)
    #
    def crop_by_center(self, image, **kargs):
        return 
    #
    def crop_by_poits(self, image, start_point, end_point, color, thickness):
  
        image = cv2.rectangle(image, start_point, end_point, color, thickness)
        return image, image[start_point[1] + thickness:end_point[1] - thickness, start_point[0] + thickness:end_point[0] - thickness]








class Interactive_Graphics():
    def __init__(self) -> None:
        self.depth_activation_threshold = None
        #white_image = np.zeros([512, 1024, 1], dtype=np.uint8)
        #white_image.fill(140)

        
        # Line thickness of 2 px
        # font, org, fontScale, color, fontThickness
        #setting_image = cv2.putText(white_image, "Settings", (50, 200), cv2.FONT_HERSHEY_SIMPLEX,
        #            5, (255, 0, 0), 2, cv2.LINE_AA)
        #cv2.imshow("setting_image", setting_image)
        #cv2.createTrackbar('slider', "setting_image", 0, 100, self.depth_change)

        cv2.namedWindow('controls', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('controls', 300, 100)
        cv2.createTrackbar('depthlevel', "controls", 0, 100, self.depth_change)




    def depth_change(self, value):
        self.depth_activation_threshold = value
        print(value)



    def get_feedback_graphics(self, depth_flag, infos):
        if depth_flag: 
            print(infos["org"])
            image = cv2.rectangle(infos["frame_original"], infos["xa"], infos["xb"], (0, 255, 0), 20)
            final = cv2.resize(image,infos["shape"])
            final = cv2.putText(final, "Ok", infos["org"], infos["font"], 
                   5, (0, 255, 0), 2, cv2.LINE_AA)
        
        else:
            print(infos["org"])
            image = cv2.rectangle(infos["frame_original"], infos["xa"], infos["xb"], (0, 255, 0), 20)
            final = cv2.resize(image, infos["shape"])
            final = cv2.circle(final, infos["org"], infos["diff"], infos["color_var__diff"], 2)

        return final





def scale_points(start_point, end_point, original_shape, new_shape):
    x_factor = original_shape[0]/new_shape[0]
    y_factor = original_shape[1]/new_shape[1]
    factors = (x_factor, y_factor)
    start_point = (int(start_point[0]/factors[0]), int(start_point[1]/factors[1]))
    end_point = (int(end_point[0]/factors[0]), int(end_point[1]/factors[1]))
    return start_point, end_point