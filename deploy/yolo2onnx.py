import argparse
import tqdm

import torch
import onnx
from network.yolo_nano_onnx import YOLONano

import datetime
parser = argparse.ArgumentParser()
#parser.add_argument("--image_folder", type=str, default="data/Ucar_test_0317", help="path to dataset")
parser.add_argument("--num_classes", type=int, default=1, help="# of classes of the dataset")
# parser.add_argument("--model_def", type=str, default="config/yolov3-ucar.cfg", help="path to model definition file")
parser.add_argument("--weights_path", type=str, default="../checkpoints/yolov3_ckpt_93.pth", help="path to weights file")
#parser.add_argument("--class_path", type=str, default="data/Ucar_data/ucar.names", help="path to class label file")
#parser.add_argument("--conf_thres", type=float, default=0.9, help="object confidence threshold")
#parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
#parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
#parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
opt = parser.parse_args()
print(opt)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = YOLONano(opt.num_classes, opt.img_size).to(device)
if opt.weights_path.endswith(".weights"):
    # Load darknet weights
    model.load_darknet_weights(opt.weights_path)
else:
    # Load checkpoint weights
    model.load_state_dict(torch.load(opt.weights_path))
model.eval()

input_shape = (3, 416, 416)
dummy_input = torch.randn(opt.batch_size, *input_shape).to(device)

model_onnx_path = 'yolo_nano.onnx'

torch.onnx.export(model, dummy_input, model_onnx_path, opset_version=11, verbose=True, export_params=True)
#onnx_model = onnx.load(model_onnx_path)
#onnx.checker.check_model(onnx_model)
