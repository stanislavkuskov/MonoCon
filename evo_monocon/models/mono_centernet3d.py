import torch.nn as nn

class EvoCenterNetMono3D(nn.Module):
    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None):
        assert bbox_head is not None
        super(EvoCenterNetMono3D, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.bbox_head = bbox_head

    def forward(self, img, img_metas):
        x = self.backbone(img)
        x = self.neck(x)
        outs = self.bbox_head(x)
        bbox_outputs = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=False)
        return bbox_outputs
