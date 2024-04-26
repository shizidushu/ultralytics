# pip install "typer[all]"
import typer
from typing import List, Optional

import os
import logging
from copy import deepcopy

import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import select_device
from ultralytics.utils.checks import check_imgsz, check_requirements
from ultralytics.utils.files import file_size
from ultralytics.nn.autobackend import check_class_names
from ultralytics.nn.tasks import PoseModel, SegmentationModel, DetectionModel
from ultralytics.nn.modules import C2f, Detect, RTDETRDecoder

logging.basicConfig(level=logging.DEBUG)

app = typer.Typer()

@app.command()
def main(model_file: str, batch: int = 1, imgsz: List[int] = [640,640], output_names: Optional[List[str]] = None, opset:int =12, dynamic:bool=False, simplify:bool =True, half:bool = False, device: str='0'):
    
    net: YOLO = YOLO(model_file)
    
    device: torch.device = select_device('cpu' if device is None else device)
    model: PoseModel = net.model
    model.names = check_class_names(model.names)
    
    if half and device.type == 'cpu':
        logging.warning('WARNING ⚠️ half=True only compatible with GPU export, i.e. use device=0')
        half = False
        assert not dynamic, 'half=True not compatible with dynamic=True, i.e. use only one.'

    imgsz = check_imgsz(imgsz, stride=model.stride, min_dim=2)    
    # Input
    im = torch.zeros(batch, 3, *imgsz).to(device)
    
    # Update model
    model = deepcopy(model).to(device)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.float()
    model = model.fuse()
    
    for k, m in model.named_modules():
        if isinstance(m, (Detect, RTDETRDecoder)):  # Segment and Pose use Detect base class
            m.dynamic = dynamic
            m.export = True
            m.format = "onnx"
        elif isinstance(m, C2f):
            # Split provides cleaner ONNX graph
            m.forward = m.forward_split

    y = None
    for _ in range(2):
        y = model(im)  # dry runs
    if half and device.type != 'cpu':
        im, model = im.half(), model.half()  # to FP16
    
    
    requirements = ['onnx>=1.12.0']
    if simplify:
        requirements += ['onnxsim>=0.4.17', 'onnxruntime-gpu' if torch.cuda.is_available() else 'onnxruntime']
    check_requirements(requirements)
    import onnx  # noqa
    
    
    f_infix = ""
    
    
    if output_names is None:
        if isinstance(model, SegmentationModel):
            output_names = ['detect_box', 'detect_cls', 'output0', 'output1']
        elif isinstance(model, PoseModel):
            output_names = ['detect_box', 'detect_cls', "pose_coord", "pose_conf"]
        elif isinstance(model, DetectionModel):
            output_names = ['detect_box', 'detect_cls']
    
    # if dynamic:
    #     # dynamic = {'images': {0: 'batch', 2: 'height', 3: 'width'}}  # shape(1,3,640,640)
    #     dynamic = {'images': {0: 'batch'}}
    #     for output_name in output_names:
    #         dynamic[output_name] = {0: 'batch'}
    
    if dynamic:
        dynamic = {'images': {0: 'batch', 2: 'height', 3: 'width'}}  # shape(1,3,640,640)
        if isinstance(model, SegmentationModel):
            raise "SegmentationModel is not supported now"
            dynamic['output0'] = {0: 'batch', 2: 'anchors'}  # shape(1, 116, 8400)
            dynamic['output1'] = {0: 'batch', 2: 'mask_height', 3: 'mask_width'}  # shape(1,32,160,160)
        elif isinstance(model, PoseModel):
            print("------------add dynamic to PoseModel----------------")
            dynamic['detect_box'] = {0: 'batch', 1: 'box', 2: 'anchors'}
            dynamic['detect_cls'] = {0: 'batch', 1: 'cls', 2: 'anchors'}
            dynamic['pose_coord'] = {0: 'batch', 1: 'points', 2: 'coord',  3: 'anchors'}
            dynamic['pose_conf'] = {0: 'batch', 1: 'points', 2: 'conf', 3: 'anchors'}
        elif isinstance(model, DetectionModel):
            print("------------add dynamic to DetectionModel----------------")
            dynamic['detect_box'] = {0: 'batch', 1: 'box', 2: 'anchors'} # (1, 4, 8400)
            dynamic['detect_cls'] = {0: 'batch', 1: 'cls', 2: 'anchors'} # (1, 80, 8400)

        
        f_infix = f_infix + "_dynamic"
    
    if half:
        f_infix = f_infix + "_float16"
        
    # output_f = os.path.splitext(model_file)[0] + f_infix + ".onnx"
    output_f = model_file.replace(".pt", f_infix + ".onnx")
    os.makedirs(os.path.abspath(os.path.join(output_f, os.pardir)), exist_ok=True)
    
    print("Using opset_version: ", opset)
    
    # https://pytorch.org/docs/master/onnx.html#module-torch.onnx
    torch.onnx.export(
        model.cpu() if dynamic else model, # --dynamic only compatible with cpu
        im.cpu() if dynamic else im,
        output_f,
        verbose=False,
        opset_version=opset,
        do_constant_folding=True,  # WARNING: DNN inference with torch>=1.12 may require do_constant_folding=False
        input_names=['images'],
        output_names=output_names,
        dynamic_axes=dynamic or None
    )
    
    # Checks
    model_onnx = onnx.load(output_f)  # load onnx model
    # onnx.checker.check_model(model_onnx)  # check onnx model
    
    if isinstance(model, PoseModel):
        print("You are exporting a PoseModel to ONNX format")
        metadata = {
            "stride": int(max(model.stride)),
            "task": model.task,
            "batch": batch,
            "imgsz": imgsz,
            "kpt_shape": model.model[-1].kpt_shape,
        }
    elif isinstance(model, DetectionModel):
        print("You are exporting a DetectionModel to ONNX format")
        metadata = {
            "stride": int(max(model.stride)),
            "task": model.task,
            "batch": batch,
            "imgsz": imgsz
        }
    
    if simplify:
        try:
            import onnxsim

            logging.info(f'ONNX: simplifying with onnxsim {onnxsim.__version__}...')
            # subprocess.run(f'onnxsim "{f}" "{f}"', shell=True)
            model_onnx, check = onnxsim.simplify(model_onnx)
            assert check, 'Simplified ONNX model could not be validated'
        except Exception as e:
            logging.info(f'ONNX: simplifier failure: {e}')
    
    # Metadata
    for k, v in metadata.items():
        meta = model_onnx.metadata_props.add()
        meta.key, meta.value = k, str(v)

    onnx.save(model_onnx, output_f)
    
    logging.info(
        f"\nSaved as '{os.path.abspath(output_f)}', ({file_size(output_f):.1f} MB)\n\n"        
    )


if __name__ == "__main__":
    app()