import torch
from ultralytics.utils.checks import check_version
from ultralytics.utils.tal import dist2bbox, make_anchors
from ultralytics.nn.modules.head import Detect, Pose

TORCH_1_10 = check_version(torch.__version__, '1.10.0')

import torch.nn as nn

def custom_make_anchors2(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing='ij') if TORCH_1_10 else torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return anchor_points, stride_tensor

original_detect_forward = Detect.forward
def custom_detect_forward(self, x):
    """Concatenates and returns predicted bounding boxes and class probabilities."""
    
    shape = x[0].shape  # BCHW
    if self.export and self.format == "onnx":
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        print("Entering custom export script of Detect")
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            # self.anchor_list, self.stride_list = custom_make_anchors2(x, self.stride, 0.5)
            self.anchor_list, self.strides_list = ([x.transpose(0, 1) for x in sublist] for sublist in custom_make_anchors2(x, self.stride, 0.5))
            self.shape = shape
        # x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        # box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        
        # dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides.view(torch.Size([1]) + self.strides.shape)
        
        box_list = []
        cls_list = []
        for i in range(self.nl):
            box_item, cls_item = x[i].view(shape[0], self.no, -1).split((self.reg_max * 4, self.nc), 1)
            box_list.append(self.dfl(box_item))
            cls_list.append(cls_item)
        
        box = torch.cat(box_list, -1)
        cls = torch.cat(cls_list, -1)
            
        dbox = dist2bbox(box, self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides.view(torch.Size([1]) + self.strides.shape)
        
        return [dbox, cls]
    
    return original_detect_forward(self, x)


original_pose_init = Pose.__init__
def custom_pose_init(self, nc=80, kpt_shape=(17, 3), ch=()):
    """Initialize YOLO network with default parameters and Convolutional Layers."""
    original_pose_init(self, nc, kpt_shape, ch)
    self.point_conv = nn.Conv2d(17, 17, 1, bias=False, groups=17).requires_grad_(False)

original_pose_forward = Pose.forward
def custom_pose_forward(self, x):
    """Perform forward pass through YOLO model and return predictions."""
    bs = x[0].shape[0]  # batch size
    #  -----------------start custom export-----------------
    if self.export and self.format == "onnx":
        print("Entering custom export script of Pose")
        kpt = [self.cv4[i](x[i]) for i in range(self.nl)]
        x = self.detect(self, x)
        regs = self.custom_kpts_decode_v4(bs, kpt)
        if isinstance(regs, list):
            for y in regs:
                x.append(y)
        else:
            x.append(regs)
        return x
    
    decode_version = 4
    
    if decode_version == 0:
        cls1, cls2, reg1, reg2, reg3 = x
        pred_kpt = self.custom_kpts_decode_v0(bs, reg1, reg2, reg3)
    elif decode_version == 1:
        cls1, cls2, reg1, reg2 = x
        pred_kpt = self.custom_kpts_decode_v1(bs, reg1, reg2)
    elif decode_version == 2:
        cls1, cls2, reg1, reg2 = x
        pred_kpt = self.custom_kpts_decode_v2(bs, reg1, reg2)
    elif decode_version == 3:
        cls1, cls2, reg1, reg2 = x
        pred_kpt = self.custom_kpts_decode_v3(bs, reg1, reg2)
    elif decode_version == 4:
        cls1, cls2, reg1, reg2 = x
        pred_kpt = self.custom_kpts_decode_v4(bs, reg1, reg2)
    else:
        raise "Unknown Keypoint Decode Version!"

    x = self.detect(self, [cls1, cls2])
    
    return torch.cat([x, pred_kpt], 1) # if self.export else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))
    
    # last resort
    return original_pose_forward(self, x)

def custom_kpts_decode_v0(self, bs, kpts):
    # 在后处理时处理关键点解码anchor和stride
    return kpts

def custom_kpts_decode_v1(self, bs, kpts):
    # 在模型中处理关键点解码anchor和stride  -- RKNN上存在问题
    kpts = torch.cat([kpts[i].view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, 17*3, h*w)
    y = kpts.view(bs, *self.kpt_shape, -1)
    reg_coords = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
    reg_confidences = y[:, :, 2:3]
    return [reg_coords, reg_confidences]

def custom_kpts_decode_v2(self, bs, kpts):
    # 在后处理时处理关键点解码anchor和stride   -- RKNN上OK
    kpts = torch.cat([kpts[i].view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, 17*3, h*w)
    y = kpts.view(bs, *self.kpt_shape, -1)
    return [y[:, :, :2], y[:, :, 2:3]]

def custom_kpts_decode_v3(self, bs, kpts):
    # 尝试在模型中处理关键点解码anchor和stride
    # 采用卷积的方式来实现简单的加法和乘法   --RKNN上报错
    # kpts: [torch.Size([1, 51, 80, 80]), torch.Size([1, 51, 40, 40]), torch.Size([1, 51, 20, 20])]
    pred_kpts = torch.cat([kpts[i].view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, 17*3, h*w)
    y = pred_kpts.view(bs, *self.kpt_shape, -1)
    reg = self.point_conv(y[:, :, :2])
    
    reg1 = self.anchor_conv1(torch.permute(reg[:,:,:1,:],[0,3,1,2]))
    reg2 = self.anchor_conv2(torch.permute(reg[:,:,1:,:],[0,3,1,2]))
    reg_coords = torch.permute(self.stride_conv(torch.cat([reg1,reg2],-1)), [0,2,3,1])
    
    reg_confidences = y[:, :, 2:3]
    return [reg_coords, reg_confidences]

def custom_kpts_decode_v4(self, bs, kpts):
    # 尝试在模型中处理关键点解码anchor和stride
    # 尝试不同的特征图分别解码        --RKNN上OK
    coords_list = []
    confidence_list = []
    for i in range(self.nl):
        kpt = kpts[i].view(bs, self.nk, -1)
        y = kpt.view(bs, *self.kpt_shape, -1)
        a = (y[:, :, :2] * 2.0 + (self.anchor_list[i] - 0.5)) * self.strides_list[i]
        coords_list.append(a)
        confidence_list.append(y[:, :, 2:3])
    return [torch.cat(coords_list, -1), torch.cat(confidence_list, -1)]