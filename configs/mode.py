model = dict(
    list_modes = ["train_det","inference_det","train_ocr","inference_ocr"],
    n_boxes = 20,
    dim = 12
)

transform = dict(

    size_resize_img = (240,240),
    ColorJitter = dict(brightness= (0.05,0.8),
                        #brightness= 0.7,
                        contrast = 0.6,
                        saturation = 0.4,
                        hue = 0.3
     ),  
    RandomPerspective = dict(distortion_scale=0,
                            p=0,
                            fill=255),

    GaussianBlur = dict(kernel_size=(9,9),
                        sigma = (1.5,1.8)))

transform_OCR = dict(

    size_resize_img = (480,640),
    ColorJitter = dict(brightness= 0,
                        #brightness= 0.7,
                        contrast = (1,2),
                        saturation = 0,
                        hue = 0
     )
)

dataset = dict(
    folder_dir = "data/DATAMATRIX/hd",
    folder_dir_save = "data/DATAMATRIX/hd/aug",
    folder_dir_save_pre = "data/DATAMATRIX/hd/predicted",
    img_path_save_pre_pad = "data/DATAMATRIX/hd/predicted_padded",
    transform = transform,
    transform_OCR = transform_OCR,
    modality = "Detector" 
)

dataloader = dict(batch_size=1,
            shuffle=False,
            sampler=None,
            batch_sampler=None,
            num_workers=0,
            collate_fn=None,
            pin_memory=False,
            drop_last=False,
            timeout=0,
            worker_init_fn=None,
            prefetch_factor=2,
            persistent_workers=False)

training = dict()





