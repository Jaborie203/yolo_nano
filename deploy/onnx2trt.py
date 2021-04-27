import os
import common
import glob
import argparse
import random
import tensorrt as trt
import numpy as np
from PIL import Image, ImageOps, ImageFile
from data_processing import PreprocessYOLO
from matplotlib.ticker import NullLocator
from utils.utils import *
import matplotlib.pyplot as plt
#from utils.datasets import *

from utils.common_funcs import to_cpu, non_max_suppression


ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.backends.cudnn.benchmark = True

no_pad2square = False
CATEGORY_NUM = 1

parser = argparse.ArgumentParser()
parser.add_argument("--image_folder", type=str, default="../data/Ucar_test_0317", help="path to dataset")
parser.add_argument("--num_classes", type=int, default=1, help="# of classes of the dataset")
# parser.add_argument("--model_def", type=str, default="config/yolov3-ucar.cfg", help="path to model definition file")
parser.add_argument("--weights_path", type=str, default="../checkpoints/yolov3_ckpt_93.pth", help="path to weights file")
parser.add_argument("--class_path", type=str, default="../data/Ucar_data/ucar.names", help="path to class label file")
parser.add_argument("--conf_thres", type=float, default=0.9, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
opt = parser.parse_args()

# From https://github.com/eriklindernoren/PyTorch-YOLOv3
class YOLOLayer(object):
    # detection layer
    def __init__(self, anchors, num_classes, img_dim=416):
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.obj_scale = 1
        self.img_dim = img_dim
        self.grid_size = 0

    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size
        g = self.grid_size
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.img_dim / self.grid_size
        # Calculate offsets for each grid
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def forward(self, x, img_dim=None):
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        self.img_dim = img_dim

        num_samples = x.size(0)
        grid_size = x.size(2)

        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes+5, grid_size, grid_size)
            .permute(0,1,3,4,2)
            .contiguous()
        )

        x = torch.sigmoid(prediction[..., 0]) # center x
        y = torch.sigmoid(prediction[..., 1]) # center y
        w = prediction[..., 2] # width
        h = prediction[..., 3] # Height
        pred_conf = torch.sigmoid(prediction[..., 4]) # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:]) # Cls Pred

        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) * self.stride,
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )
        return output


anchors52 = [[10,13], [16,30], [33,23]] # 52x52
anchors26 = [[30,61], [62,45], [59,119]] # 26x26
anchors13 = [[116,90], [156,198], [373,326]] # 13x13
YOLOLayer52 = YOLOLayer(anchors52,CATEGORY_NUM,img_dim=416)
YOLOLayer26 = YOLOLayer(anchors26,CATEGORY_NUM,img_dim=416)
YOLOLayer13 = YOLOLayer(anchors13,CATEGORY_NUM,img_dim=416)


def load_image(path=''):
    with open(path,'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def preprocess(input):
    w,h = input.size

    if not no_pad2square:
        if w == h:
            image = input
        else:
            dim_diff = abs(w - h)
            padding_1,padding_2 = dim_diff // 2,dim_diff - dim_diff // 2
            padding = (0,padding_1,0,padding_2) if w > h else (padding_1,0,padding_2,0)
            image = ImageOps.expand(input,border=padding,fill=0)  ##left,top,right,bottom
    else:
        image = input

    image = image.resize((416,416),Image.ANTIALIAS)

    input_data = np.array(image).astype(np.float32)
    input_data = input_data/255.0
    input_data = np.transpose(input_data,(2,0,1))
    input_data = np.expand_dims(input_data, axis=0)
    return input_data


def postprecess(outputs):
    detections = []
    temp = YOLOLayer52.forward(torch.tensor(outputs[0].reshape(-1, 18, 52, 52)),img_dim=416)
    detections.append(temp)
    temp = YOLOLayer26.forward(torch.tensor(outputs[1].reshape(-1, 18, 26, 26)),img_dim=416)
    detections.append(temp)
    temp = YOLOLayer13.forward(torch.tensor(outputs[2].reshape(-1, 18, 13, 13)),img_dim=416)
    detections.append(temp)

    detections = to_cpu(torch.cat(detections,1))
    return detections


TRT_LOGGER = trt.Logger()

def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 30 # 1GB
            builder.max_batch_size = 1
            #if builder.platform_has_fast_fp16:
                #builder.fp16_mode = True
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                parser.parse(model.read())
            print('Completed parsing of ONNX file')

            #last_layer = network.get_layer(network.num_layers-1)
            #if not last_layer.get_output(0):
            #    network.mark_output(last_layer.get_output(0))
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()

def main():

    """Create a TensorRT engine for ONNX-based YOLOv3-608 and run inference."""

    # Try to load a previously generated YOLOv3-608 network graph in ONNX format:
    onnx_file_path = 'yolo_nano.onnx'
    engine_file_path = "yolo_nano.trt"
    input_image_path = '../data/Ucar_test_F01'
    files = sorted(glob.glob(input_image_path + '/*.jpg'))
    # Two-dimensional tuple with the target network's (spatial) input resolution in HW ordered
    input_resolution_yolov3_HW = (416, 416)

    # Create a pre-processor object by specifying the required input resolution for YOLOv3
    preprocessor = PreprocessYOLO(input_resolution_yolov3_HW)

    #input_image = load_image(input_image_path)
    #img_input = preprocess(input_image)
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    classes = load_classes(opt.class_path)
    # Do inference with TensorRT
    with get_engine(onnx_file_path, engine_file_path) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        # fps = []
        for file in files:
             # Load an image from the specified input path, and return it together with  a pre-processed version
            image_raw, image = preprocessor.process(file)
            # Do inference
            print('Running inference on image {}...'.format(file))
            # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
            inference_start = time.time()
            inputs[0].host = image
            trt_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            inference_end = time.time()
            inference_time = inference_end-inference_start
            print('inference time : %f, FPS: %f' % (inference_time, 1 / inference_time))
            # fps.append(1 / inference_time)

            yolo_start = time.time()
            detections = postprecess(trt_outputs)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)[0] #1 x n x 7
            #print(detections)
            image_raw = np.array(image_raw)
            height, width, _ = image_raw.shape
            plt.figure()
            fig, ax = plt.subplots(1)
            ax.imshow(image_raw)

            if detections is not None:
                detections = rescale_boxes(detections, opt.img_size, image_raw.shape[:2])
                unique_labels = detections[:, -1].cpu().unique()
                unique_conf = detections[:, -3].cpu().unique()
                max_conf = max(unique_conf)
                n_cls_preds = len(unique_labels)
                bbox_colors = random.sample(colors, n_cls_preds)

                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    # if(max_conf == conf):
                    print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
                    box_w = x2 - x1
                    box_h = y2 - y1

                    center_x = (x1 + x2) / 2.0
                    center_y = (y1 + y2) / 2.0
                    if (center_y > height / 2):
                        filename0 = file.split("/")[-1].split(".")[0]
                        print(
                            [f"{filename0}.png", center_x.item(), center_y.item(), box_w.item(), box_h.item()])

                        color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                        bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color,
                                                  facecolor="none")
                        ax.add_patch(bbox)
                        plt.text(
                            x1,
                            y1,
                            # s=classes[int(cls_pred)],
                            s=conf.item(),
                            color="white",
                            verticalalignment="top",
                            bbox={"color": color, "pad": 0},
                        )

            plt.axis("off")
            plt.gca().xaxis.set_major_locator(NullLocator())
            plt.gca().yaxis.set_major_locator(NullLocator())
            filename = file.split("/")[-1].split(".")[0]
            plt.savefig(f"../output_trt/{filename}.png", bbox_inches="tight", pad_inches=0.0)
            # plt.show()
            plt.close()
            yolo_end = time.time()
            yolo_time = yolo_end-yolo_start
            print('yolo time : %f' % (yolo_time))
            print('all time : %f' % (yolo_end-inference_start))
        # plt.plot(fps)
        # plt.show()
        # print(np.array(fps).mean())
if __name__ == '__main__':
    main()