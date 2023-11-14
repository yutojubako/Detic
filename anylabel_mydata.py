import argparse
import glob
import json
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
from tqdm import tqdm
import sys
import mss

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

sys.path.insert(0, 'third_party/CenterNet2/')
from centernet.config import add_centernet_config
from detic.config import add_detic_config

from detic.predictor import VisualizationDemo

# ディレクトリ内のすべての画像に対して推論を実行


image_dir = '1_20231030'
image_files = glob.glob(os.path.join(image_dir, '**', '*.png'), recursive=True)
image_files += glob.glob(os.path.join(image_dir, '**', '*.jpg'), recursive=True)

def setup_cfg(args):
    cfg = get_cfg()
    if args.cpu:
        cfg.MODEL.DEVICE="cpu"
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand' # load later
    if not args.pred_all_class:
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", help="Take inputs from webcam.")
    parser.add_argument("--cpu", action='store_true', help="Use CPU only.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--vocabulary",
        default="lvis",
        choices=['lvis', 'openimages', 'objects365', 'coco', 'custom'],
        help="",
    )
    parser.add_argument(
        "--custom_vocabulary",
        default="",
        help="",
    )
    parser.add_argument("--pred_all_class", action='store_true')
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False

def get_image_files(image_dir):
    image_files = glob.glob(os.path.join(image_dir, '**', '*.png'), recursive=True)
    image_files += glob.glob(os.path.join(image_dir, '**', '*.jpg'), recursive=True)
    return image_files

def save_results(predictions, class_names, img, image_file, image_dir, output_dir, start_dir):
    result = {
        "version": "0.3.3",
        "flags": {},
        "shapes": [],
        "imagePath": ".." + os.path.relpath(image_file, start_dir).replace(image_dir, '', 1),
        "imageData": None,
        "imageHeight": img.shape[0],
        "imageWidth": img.shape[1],
        "text": ""
    }

    for i in range(len(predictions["instances"])):
        bbox = predictions["instances"].pred_boxes.tensor[i].tolist()
        label = class_names[predictions["instances"].pred_classes[i].item()]
        all_labels_set.add(label)
        shape = {
            "label": label,
            "text": "",
            "points": [[bbox[0], bbox[1]], [bbox[2], bbox[3]]],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {}
        }
        result["shapes"].append(shape)

    output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(image_file))[0] + '.json')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=4)

def save_visualized_output(visualized_output, image_file, output_dir):
    output_image_path = os.path.join(output_dir, os.path.basename(image_file))
    cv2.imwrite(output_image_path, visualized_output.get_image()[:, :, ::-1])

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    image_dir = '2_20231114'

    # VisualizationDemoクラスのインスタンスを作成
    demo = VisualizationDemo(cfg, args)

    from detectron2.data import MetadataCatalog
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    class_names = metadata.thing_classes

    # 推論を実行し、結果を保存
    output_dir = os.path.join(image_dir, 'annotations')
    os.makedirs(output_dir, exist_ok=True)

    output_image_dir = os.path.join(image_dir, 'images_detic')
    os.makedirs(output_image_dir, exist_ok=True)

    all_labels_set = set()
    start_dir = os.getcwd()

    os.makedirs(output_dir, exist_ok=True)
    image_files = get_image_files(image_dir)

    for image_file in tqdm(image_files):
        img = read_image(image_file, format="BGR")
        predictions, visualized_output = demo.run_on_image(img)
        save_results(predictions, class_names, img, image_file, image_dir, output_dir, start_dir)
        save_visualized_output(visualized_output, image_file, output_image_dir)

# 最後に、すべてのラベルを表示
print(list(all_labels_set))
