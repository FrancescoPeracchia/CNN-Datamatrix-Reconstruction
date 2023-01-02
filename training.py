#from distutils.log import info
from pickletools import read_uint1
import logging
from dataset import DMdataset
from torch.utils.data import DataLoader
import argparse
from mmcv import Config
from tools.loop import Tloop
from models.model import Recostructor
def run(cfg):

    #create models
    logging.info("Creating Model ... %s ",cfg.model)

    #dataset
    dataset_train = DMdataset(cfg.dataset,"train")

    #dataloader
    dataloader_train = DataLoader(dataset_train, batch_size=cfg.dataloader.batch_size, shuffle=cfg.dataloader.shuffle, sampler=cfg.dataloader.sampler,
           batch_sampler=cfg.dataloader.batch_sampler, num_workers=cfg.dataloader.num_workers, collate_fn=cfg.dataloader.collate_fn,
           pin_memory=cfg.dataloader.pin_memory, drop_last=cfg.dataloader.drop_last, timeout=cfg.dataloader.timeout,
           worker_init_fn=cfg.dataloader.worker_init_fn,prefetch_factor=cfg.dataloader.prefetch_factor,
           persistent_workers=cfg.dataloader.persistent_workers)

    

    model = Recostructor(cfg.model.n_boxes,cfg.model.dim)

    _ = 0
    Tloop(_,model,dataloader_train)
    


    
    #launch training loop
    logging.info("")
    print("BYE")

    #logging info





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