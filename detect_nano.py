from __future__ import division

from network import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import csv
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import cv2
import copy
from matplotlib.ticker import NullLocator

from PIL import Image, ImageDraw
    # as pil
# import PIL as pil

from network.yolo_nano_network import YOLONano
from opt import opt

import datetime

def print_model_param_nums(model):
    total = sum([param.nelement() if param.requires_grad else 0 for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/Ucar_test_F01", help="path to dataset")
    parser.add_argument("--num_classes", type=int, default=1, help="# of classes of the dataset")
    # parser.add_argument("--model_def", type=str, default="config/yolov3-ucar.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/yolov3_ckpt_93.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/Ucar_data/ucar.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.9, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("output", exist_ok=True)

    model = YOLONano(opt.num_classes, opt.img_size).to(device)
    print_model_param_nums(model)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))
    model.eval()
    dataloader = DataLoader(
        ImageFolder(opt.image_folder, img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu
    )
    classes = load_classes(opt.class_path)
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor()
    imgs = []
    img_detections = []
    print("\nPerforming object detection:")
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        input_imgs = input_imgs.to(device)
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
            print(detections)
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time-prev_time)
        fps = 1.0 / (current_time - prev_time)
        prev_time = current_time

        print("\t+ Batch %d, Inference Time: %s, FPS: %f f/s" % (batch_i, inference_time, fps))

        imgs.extend(img_paths)
        img_detections.extend(detections)

    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    print("\nSaving images:")

    with open("output/result.csv", 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile)
        for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
            print("(%d) Image: '%s'" % (img_i, path))

            img = np.array(Image.open(path))
            height, width, _ = img.shape
            plt.figure()
            fig, ax = plt.subplots(1)
            ax.imshow(img)

            if detections is not None:
                detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
                unique_labels = detections[:, -1].cpu().unique()
                unique_conf = detections[:,-3].cpu().unique()
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
                    if(center_y > height / 2):
                        filename0 = path.split("/")[-1].split(".")[0]
                        writer.writerow(
                            [f"{filename0}.png", center_x.item(), center_y.item(), box_w.item(), box_h.item()])

                        color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                        bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
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
            filename = path.split("/")[-1].split(".")[0]
            plt.savefig(f"output/{filename}.png", bbox_inches="tight", pad_inches=0.0)
            # plt.show()
            plt.close()
