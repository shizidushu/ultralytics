

## 创建环境
```bash
conda create --name yolov8_rknn_dev python=3.9
conda activate yolov8_rknn_dev
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install albumentations onnxruntime onnxruntime-gpu onnxsim


pip install -e .

# 训练完成之后导出ONNX模型
export_onnx_for_rknn last.pt
```