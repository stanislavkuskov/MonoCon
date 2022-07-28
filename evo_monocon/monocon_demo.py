import torch
from argparse import ArgumentParser
from os import path as osp
import mmcv

from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms

from configs.kitti_mono3d_3class_monocon import data as cfg_data

from datasets.monocon_dataset import MonoConEvalDataset

from models.dla import DLA
from models.dlaup import DLAUp
from models.monocon_head_inference import MonoConHeadInference
from models.mono_centernet3d import EvoCenterNetMono3D

from utils.show_result import draw_camera_bbox3d_on_img
from utils.cam_box3d import CameraInstance3DBoxes

def init_model(config, checkpoint=None, device='cuda:0'):
    """Initialize a model from config file, which could be a 3D detector or a
    3D segmentor.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str): Device to use.

    Returns:
        nn.Module: The constructed detector.
    """

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

def inference_mono_3d_detector(model, test_dataloader):
    """Inference image with the monocular 3D detector.

    Args:
        model (nn.Module): The loaded detector.
        image (str): Image files.
        ann_file (str): Annotation files.

    Returns:
        tuple: Predicted results and data from pipeline.
    """
    
    device = next(model.parameters()).device  # get device for input data from model

    data = {}
    test_data = next(iter(test_dataloader))
    data['img'] = test_data[0].to(device)
    
    meta = test_data[1]
    # Dict of lists into list of dicts
    data['img_metas'] = [
        {
            key:value[index] for key,value in meta.items()
        } for index in range(max(map(len,meta.values())))
    ]

    with torch.no_grad():
        result = model(data['img'], data['img_metas'])

    return result, data


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
    
    # TODO make torch.dataset for image
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

    # test a single image
    (batch_det_scores, batch_det_bboxes_3d, batch_labels), data = \
        inference_mono_3d_detector(
            model,
            test_dataloader
        )
    # TODO Переписать
    # box_type_3d = img_metas[0]['box_type_3d']
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
    file_name = osp.split(img_filename)[-1].split('.')[0]

    # read from file because img in data_dict has undergone pipeline transform
    img = mmcv.imread(img_filename)

    res = draw_camera_bbox3d_on_img(
        bboxes3d=det_results[0][0], raw_img=img,
        cam_intrinsic=data['img_metas'][0]['cam_intrinsic']
    )
    import cv2
    cv2.imwrite("res.jpg",res)

if __name__ == '__main__':
    main()
