import torch
from argparse import ArgumentParser
from copy import deepcopy
from os import path as osp

import mmcv
from mmcv.parallel import collate, scatter

from models.dla import DLA
from models.dlaup import DLAUp
from models.monocon_head_inference import MonoConHeadInference
from models.mono_centernet3d import CenterNetMono3D

from mmdet3d.datasets.pipelines import Compose
from mmdet3d.core.bbox import get_box_type

from mmdet3d.core import Box3DMode
from configs.kitti_mono3d_3class_monocon import data as cfg_data

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
    #TODO make model from classes
    # model = CenterNetMono3D()
    args = config.model.copy()

    backbone = DLA(34)
    neck = DLAUp()
    
    head_args= args["bbox_head"]
    del head_args["type"]
    head_args["test_cfg"] = config.model.test_cfg
    head = MonoConHeadInference(**head_args)

    model = CenterNetMono3D(backbone, neck, head)

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
    # build the data pipeline
    # print("__x___"*10)
    # print(cfg.data.test.pipeline)
    # print("______"*10)
    test_pipeline = deepcopy(cfg.data.test.pipeline)
    test_pipeline = Compose(test_pipeline)
    # print(test_pipeline)
    # print("__x___"*10)
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
    else:
        # this is a workaround to avoid the bug of MMDataParallel
        data['img_metas'] = data['img_metas'][0].data
        data['img'] = data['img'][0].data

    # print(data['img_metas'])
    # forward the model
    # print(data.keys())
    with torch.no_grad():
        print(data['img_metas'])
        result = model(return_loss=False, rescale=True, **data)
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
    # show the results
    print(result)
    # [{'img_bbox': {'boxes_3d': CameraInstance3DBoxes(
    # tensor([[1.9634, 1.5211, 9.0842, 1.2410, 1.9338, 0.5110, 0.0898]])), 'scores_3d': tensor([18.9357]), 'labels_3d': tensor([0])}}]

    ## Show results from mmdetection
    # show_result_meshlab(
    #     data,
    #     result,
    #     args.out_dir,
    #     args.score_thr,
    #     show=args.show,
    #     snapshot=args.snapshot,
    #     task='mono-det')


if __name__ == '__main__':
    main()
