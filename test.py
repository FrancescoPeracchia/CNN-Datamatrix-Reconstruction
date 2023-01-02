from PIL import Image
from pylibdmtx.pylibdmtx import decode
from models.model import Recostructor
from dataset import DMdataset
import argparse
from mmcv import Config
import numpy as np

def run(cfg):
    cfg_eval = True
    model = Recostructor(cfg.model.n_boxes,cfg.model.dim,cfg_eval)
    dataset_train = DMdataset(cfg.dataset,"train")

    model.model_rec.eval()
    model.model_edg.eval()



    data = {}




    #path ="/home/mv/Desktop/DM_reco/DM_reco/data/DATAMATRIX/hd/aug/image_1.png"
    #shape = (20,20)

    path = "saved.png"
    shape = (18,18)
    data = dataset_train.prepare_image(path,shape)
    



    data["shape"] = shape
    data["path_output"] = "pre.png"
    
    print(data["image"].size())
    output = model.build(input = data , mode = "inference")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="path to config file")
    return parser.parse_args()

def main(args):
    cfg = Config.fromfile(args.config)
    run(cfg)


if __name__ == "__main__":
    args = parse_args()
    main(args)
