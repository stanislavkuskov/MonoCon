import cv2
import mmcv
import torch
import torchvision.transforms as transforms

from argparse import ArgumentParser
from os import path as osp
from torch.utils.data.dataloader import DataLoader

from configs.kitti_mono3d_3class_monocon import data as cfg_data

from datasets.monocon_dataset import MonoConEvalDataset

from models.dla import DLA
from models.dlaup import DLAUp
from models.monocon_head_inference import MonoConHeadInference
from models.mono_centernet3d import EvoCenterNetMono3D

from utils.show_result import draw_camera_bbox3d_on_img
from utils.cam_box3d import CameraInstance3DBoxes

def init_model(config, checkpoint=None, device='cuda:0'):
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.data = cfg_data
    config.model.pretrained = None
    config.model.train_cfg = None

    args = config.model.copy()

    backbone = DLA(34)
    neck = DLAUp()
    head_args= args["bbox_head"]
    del head_args["type"]
    
    head_args["test_cfg"] = config.model.test_cfg
    head = MonoConHeadInference(**head_args)

    model = EvoCenterNetMono3D(backbone, neck, head)

    if checkpoint is not None:
        model_ckpt = torch.load(checkpoint)
        model.load_state_dict(model_ckpt['state_dict'], strict=True)
        model.CLASSES = ['Pedestrian', 'Cyclist', 'Car']

    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model

def main():
    parser = ArgumentParser()
    parser.add_argument('image', help='image file')
    parser.add_argument('ann', help='ann file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.15, help='bbox score threshold')
    parser.add_argument(
        '--out-dir', type=str, default='demo', help='dir to save results')
    parser.add_argument(
        '--show', action='store_true', help='show online visuliaztion results')
    parser.add_argument(
        '--snapshot',
        action='store_true',
        help='whether to save online visuliaztion results')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    
    # make torch.dataset for image and metadata
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4915, 0.4823, 0.4468),
            (0.2470, 0.2435, 0.2616)
        ),
    ])

    monocon_dataset = MonoConEvalDataset(
        "../mmdetection3d-0.14.0/data/kitti/training/image_2/",
        "../mmdetection3d-0.14.0/data/kitti/kitti_infos_trainval_mono3d.coco.json",
        normalize
    )
    test_dataloader = DataLoader(
        monocon_dataset, 
        batch_size=1, 
        shuffle=False,
        num_workers=0,
    )

    # Make data for model inference
    data = {}
        # return one sample from dataloader - tiple(image, metadata)
    test_data = next(iter(test_dataloader))
    data['img'] = test_data[0].to(args.device)
    
    meta = test_data[1]
        # Dict of lists into list of dicts
    data['img_metas'] = [
        {
            key:value[index] for key,value in meta.items()
        } for index in range(max(map(len,meta.values())))
    ]

    # Model inference
    with torch.no_grad():
        (batch_det_scores, batch_det_bboxes_3d, batch_labels) = model(data['img'], data['img_metas'])

    box_type_3d = CameraInstance3DBoxes
    det_results = [
        [
            box_type_3d(
                batch_det_bboxes_3d,
                box_dim=7,
                origin=(0.5, 0.5, 0.5)
            ),
            batch_det_scores[:, -1],
            batch_labels,
        ]
    ]
    img_filename = data['img_metas'][0]['filename']

    # read from file because img in data_dict has undergone pipeline transform
    img = mmcv.imread(img_filename)
    print(data["img"].shape)
    
    img = cv2.resize(img,(1248, 384))

    res = draw_camera_bbox3d_on_img(
        bboxes3d=det_results[0][0], raw_img=img,
        cam_intrinsic=data['img_metas'][0]['cam_intrinsic']
    )

    cv2.imwrite("res.jpg",res)

if __name__ == '__main__':
    main()
