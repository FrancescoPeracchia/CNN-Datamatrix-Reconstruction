#from distutils.log import info
import sys
import os

# later for GUI
#from matplotlib.pyplot import flag
#from PyQt5 import QtWidgets
#from PyQt5.QtWidgets import QApplication, QApplication
#from PySide6.QWidgets import QApplication, QMainWindow

import cv2

import numpy as np

from imageprocessing import ImageProcessing as imgproc
import cameras as cams


# Others
#import threading
import argparse
from pathlib import Path

#midas
from models.DepthModel import MiDas
from models.BarcodeReader import Reader
#from models.Ocr import OCR
from utils.miscellaneous import HZ, Cropper, Interactive_Graphics, scale_points


def run(
        resizefactor,
        savevid,
        debug,
        camera,
):

    # Camera selection
    if camera == "standardcam":
        try:
            cam = cams.StandardCamera(camidx=0)
        except:
            print("Cannot create Object for a Standard Camera")

    elif camera == "idscam":
        try:
            cam = cams.IdsCamera()
        except:
            print("Cannot create Object for a IDS Camera")

    elif camera == "oakcam":
        try:
            cam = cams.OakCamera()
        except:
            print("Cannot create Object for a Oak Camera")

    else:
        cam = None



    if savevid:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter("./videos/test_video.mp4", fourcc, 20.0, (cam.height, cam.width))



    # Create Objects for models
    # Depth
    depth_model = MiDas("DPT_Hybrid")
    # OCR
    #ocr_model = OCR("ocrModel")

    graphic = Interactive_Graphics()
    frequency_reader = HZ(average_window=4)
    reader = Reader("./models/BarcodeDetection/saved/mode.pt")
    cropper = Cropper("AB")



    while True:
        frequency_reader.start()

        """
        Cam  workflow
        the original image is reshaped in order to be used as input to the depth model
        """

        frame_original = cam.getframe()
        if frame_original is None:
            continue

        if debug:
            cv2.imshow("frame_original", frame_original)

            
        resized = tuple(int(resizefactor * elem) for elem in (cam.width, cam.height))
        frame_resized = cv2.resize(frame_original, resized)
        if debug:
            cv2.imshow("frame_resized", frame_resized)


        if savevid:
            writer.write(frame_original)



        """
        Depth workflow :
        .generate the depth
        .interpolate and scale (it in range [0,255]) the final output result
        .show it in gray scale

        note : shape image in input has (1280, 960) as input to the resize function while interpolation reshape has (960, 1280)
               this is because they ask for different input order

        """
        reshaped = (resized[1], resized[0])
        depth_map = depth_model.get_depth(frame_resized, reshaped)

        if debug:
            cv2.imshow("depth_map", depth_map)


        

        """
        create window
        args :
        start_point = (Xa,Ya), end_point = (Xb,Yb)
        color = (Blue, Green, Red) in BGR
        thickness in pixels
        """
        #Note that the input of out Barcode Detector is  H = 480 W = 640
        
        #W is  xa[0] - xb[0]
        #H is  xa[1] - xb[1]
        t = 4
        w = 640
        h = 480
        xa = (int(resized[0] * 0.5), int(resized[1] * 0.5))
        xb = (xa[0] + w + t, xa[1] + h + t)
        image_feedback, cropped_image = cropper.crop(frame_original, start_point=xa, end_point=xb, color=(255, 255, 0), thickness=t)
        if debug:
            cv2.imshow("Cropped", cropped_image)
            cv2.imshow("Feedback", image_feedback)


        """
        create relative window in depth window
        args :
        original_shape
        depth_shape
        start_point = (Xa,Ya), end_point = (Xb,Yb)
        """

        #xa_scaled, xb_scaled = scale_points(xa, xb, original_shape, new_shape)
        xa_scaled, xb_scaled = scale_points(xa, xb, (cam.width, cam.height), resized)
        image_feedback_depth, cropped_image_depth = cropper.crop(depth_map, start_point=xa_scaled, end_point=xb_scaled, color=(255, 255, 0), thickness=6)

        """
        depth info usage
        args :
        original_shape
        depth_shape
        start_point = (Xa,Ya), end_point = (Xb,Yb)
        """

        if debug:
            cv2.imshow("Feedback depth", image_feedback_depth)
        
        depth_distance = np.mean(cropped_image_depth)
        depth_distance = 100 * (255/depth_distance - 1)
        




        """
        Activation check 
        """
        depth_activation_threshold = graphic.depth_activation_threshold
        if depth_activation_threshold == None:
            depth_activation_threshold = 0

        
        if depth_distance < depth_activation_threshold:
            depth_flag = True
        else:
            depth_flag = False
        

        


        """
        User Graphics 
        """
        diff = abs(depth_activation_threshold - depth_distance)

        infos = {}
        infos["frame_original"] = frame_original
        infos["xa"] = xa
        infos["xb"] = xb
        infos["shape"] = resized 
        infos["org"] = (150, 150)
        infos["font"] = cv2.FONT_HERSHEY_SIMPLEX
        infos["diff"] = int(diff)
        infos["color_var__diff"] = (0, 255 - diff, 150 + diff)

        final = graphic.get_feedback_graphics(depth_flag, infos)

        if debug:
            cv2.imshow("final", final)
        

        # show as image
        imgArray = ([frame_original, image_feedback], [image_feedback_depth, final])
        cv2.imshow("overview", imgproc.stackImages(0.35, imgArray))

        cv2.imshow("frame_original", frame_original)
        cv2.imshow("image_feedback_depth", image_feedback_depth)




        hz = frequency_reader.end()
        org = (50, 50)
        cropped_image_depth = cv2.putText(cropped_image_depth, str(hz), org, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)
        

        name_window = "Extracted Depth"
        cv2.imshow(name_window, cropped_image_depth)




        """
        BAR code extraction

        1. Barcode BBOX detection
        2. 
        """
        if depth_flag:
            pil_image, skip = reader.read(cropped_image)
            if skip == True:
                pass
            else:
                open_cv_image = np.array(pil_image) 
                # Convert RGB to BGR 
                open_cv_image = open_cv_image[:, :, ::-1].copy() 
                name_window = "Barcode Detection"
                cv2.imshow(name_window, open_cv_image)
                #cv2.imshow("OCR output",ocr_model.readtext(pil_image))
            



        """
        Exit condition
        Save condition


        press q from keyboard
        press t from keyboard
        """
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            cam.close()

        if cv2.waitKey(1) & 0xFF == ord('t'):
            cv2.imwrite("Cropped Image.jpg", cropped_image)


    if savevid:
        writer.release()


    cv2.destroyAllWindows()




def parse_opt():
    parser = argparse.ArgumentParser()
    #parser.add_argument("-H", "--height", type=int, default=2160, help="Output Height")
    #parser.add_argument("-W", "--width", type=int, default=3840, help="Output Width")
    #parser.add_argument("-OAI", "--openai", type=bool, default=False, help="Define usage of OAK Cam")
    parser.add_argument("-SAVEVID", "--savevid", type=bool, default=False, help="Record Video")
    parser.add_argument("-RSFCTR", "--resizefactor", type=float, default=0.6, help="Resize Factor")
    parser.add_argument("-DEBUG", "--debug", type=bool, default=False, help="Show All Pipeline Steps")
    parser.add_argument("-CAM", "--camera", type=str, default="standardcam", help="Define Camera [standardcam, idscam, oakcam]")
    opt = parser.parse_args()
    return opt

def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)