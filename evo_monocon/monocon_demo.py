import torch
from argparse import ArgumentParser
from copy import deepcopy
from os import path as osp
import mmcv
from mmcv.parallel import collate, scatter

from models.dla import DLA
from models.dlaup import DLAUp
from models.monocon_head_inference import MonoConHeadInference
from models.mono_centernet3d import EvoCenterNetMono3D

from mmdet3d.datasets.pipelines import Compose
from mmdet3d.core.bbox import get_box_type

from mmdet3d.core import Box3DMode
from configs.kitti_mono3d_3class_monocon import data as cfg_data
from utils.show_result import draw_camera_bbox3d_on_img

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
    # convert_SyncBN(config.model)
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

def inference_mono_3d_detector(model, image, ann_file):
    """Inference image with the monocular 3D detector.

    Args:
        model (nn.Module): The loaded detector.
        image (str): Image files.
        ann_file (str): Annotation files.

    Returns:
        tuple: Predicted results and data from pipeline.
    """
    
    cfg = model.cfg
    device = next(model.parameters()).device  # get device for input data from model
    test_pipeline = deepcopy(cfg.data.test.pipeline)
    test_pipeline = Compose(test_pipeline)
    box_type_3d, box_mode_3d = get_box_type(cfg.data.test.box_type_3d)
    # get data info containing calib
    data_infos = mmcv.load(ann_file)
    # find the info corresponding to this image
    for x in data_infos['images']:
        if osp.basename(x['file_name']) != osp.basename(image):
            continue
        img_info = x
        break
    data = dict(
        img_prefix=osp.dirname(image),
        img_info=dict(filename=osp.basename(image)),
        box_type_3d=box_type_3d,
        box_mode_3d=box_mode_3d,
        img_fields=[],
        bbox3d_fields=[],
        pts_mask_fields=[],
        pts_seg_fields=[],
        bbox_fields=[],
        mask_fields=[],
        seg_fields=[])

    # camera points to image conversion
    if box_mode_3d == Box3DMode.CAM:
        data['img_info'].update(dict(cam_intrinsic=img_info['cam_intrinsic']))


    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device.index])[0]

    data['img_metas'] = data['img_metas'][0]
    data['img'] = data['img'][0].data

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
    # test a single image
    result, data = inference_mono_3d_detector(model, args.image, args.ann)
    img_filename = data['img_metas'][0]['filename']
    file_name = osp.split(img_filename)[-1].split('.')[0]

    # read from file because img in data_dict has undergone pipeline transform
    img = mmcv.imread(img_filename)
    res = draw_camera_bbox3d_on_img(
        bboxes3d=result[0][0], raw_img=img,
        cam_intrinsic=data['img_metas'][0]['cam_intrinsic']
    )
    import cv2
    cv2.imwrite("res.jpg",res)
    print(type(res))

if __name__ == '__main__':
    main()
