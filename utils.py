import argparse
import onnx
from transforms import *
from PIL import Image, ImageDraw, ImageFont
import math


def str2bool(v):
    return v.lower() in ("true", "t", "1")


def parse_args():
    parser = argparse.ArgumentParser()
    # General params
    parser.add_argument("--det_model_dir", type=str, help="Path to onnx detection model weights")
    parser.add_argument("--cls_model_dir", type=str, help="Path to onnx angle classifier model weights")
    parser.add_argument("--rec_model_dir", type=str, help="Path to onnx text recognition model weights")
    parser.add_argument("-i", "--image_path", type=str, help="Path to test image")
    parser.add_argument("--onnx_check", action="store_true", help="Pass if want to check onnx model with onnx.checker")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--run_detection", action="store_true", help="Run detection model")
    group.add_argument("--run_pipeline", action="store_true", help="Run full recognition pipeline")
    # Detection params
    parser.add_argument("--det_db_thresh", type=float, default=0.3)
    parser.add_argument("--det_db_box_thresh", type=float, default=0.6)
    parser.add_argument("--det_db_unclip_ratio", type=float, default=1.5)
    parser.add_argument("--max_batch_size", type=int, default=10)
    parser.add_argument("--use_dilation", type=bool, default=False)
    # Angle classification params
    parser.add_argument("--use_angle_cls", action="store_true")
    parser.add_argument("--label_list", type=list, default=['0', '180'])
    parser.add_argument("--cls_batch_num", type=int, default=6)
    parser.add_argument("--cls_thresh", type=float, default=0.9)
    # Recognizer params
    parser.add_argument("--rec_batch_num", type=int, default=6)
    parser.add_argument(
        "--rec_char_dict_path",
        type=str,
        default="./rec_char_dict/ppocr_keys_v1.txt")
    parser.add_argument("--use_space_char", type=str2bool, default=True)
    parser.add_argument(
        "--vis_font_path", type=str, default="./fonts/latin.ttf")
    parser.add_argument("--drop_score", type=float, default=0.5)

    return parser.parse_args()


## ONNX utils

def model_check(weights_path):
    print('Checking the model!')
    onnx_model = onnx.load(weights_path)
    onnx.checker.check_model(onnx_model)
    print('The model is checked!')


## Transforms utils

def create_operators(op_param_list, global_config=None):
    """
    create operators based on the config

    Args:
        params(list): a dict list, used to create some operators
    """
    assert isinstance(op_param_list, list), 'operator config should be a list'
    ops = []
    for operator in op_param_list:
        assert isinstance(operator,
                          dict) and len(operator) == 1, "yaml format error"
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]
        if global_config is not None:
            param.update(global_config)
        op = eval(op_name)(**param)
        ops.append(op)
    return ops


def transform(data, ops=None):
    """ transform """
    if ops is None:
        ops = []
    for op in ops:
        data = op(data)
        if data is None:
            return None
    return data


## Visualization utils 

def draw_text_boxes(dt_boxes, image):
    for box in dt_boxes:
        box = np.array(box).astype(np.int32).reshape(-1, 2)
        cv2.polylines(image, [box], True, color=(255, 255, 0), thickness=2)
    return image


def draw_ocr_box_txt(image,
                     boxes,
                     txts,
                     scores=None,
                     drop_score=0.5,
                     font_path="./doc/simfang.ttf"):
    h, w = image.height, image.width
    img_left = image.copy()
    img_right = Image.new('RGB', (w, h), (255, 255, 255))

    import random

    random.seed(0)
    draw_left = ImageDraw.Draw(img_left)
    draw_right = ImageDraw.Draw(img_right)
    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        if scores is not None and scores[idx] < drop_score:
            continue
        color = (random.randint(0, 255), random.randint(0, 255),
                 random.randint(0, 255))
        draw_left.polygon(box, fill=color)
        draw_right.polygon(
            [
                box[0][0], box[0][1], box[1][0], box[1][1], box[2][0],
                box[2][1], box[3][0], box[3][1]
            ],
            outline=color)
        box_height = math.sqrt((box[0][0] - box[3][0]) ** 2 + (box[0][1] - box[3][
            1]) ** 2)
        box_width = math.sqrt((box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][
            1]) ** 2)
        if box_height > 2 * box_width:
            font_size = max(int(box_width * 0.9), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            cur_y = box[0][1]
            for c in txt:
                char_size = font.getsize(c)
                draw_right.text(
                    (box[0][0] + 3, cur_y), c, fill=(0, 0, 0), font=font)
                cur_y += char_size[1]
        else:
            font_size = max(int(box_height * 0.8), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            draw_right.text(
                [box[0][0], box[0][1]], txt, fill=(0, 0, 0), font=font)
    img_left = Image.blend(image, img_left, 0.5)
    img_show = Image.new('RGB', (w * 2, h), (255, 255, 255))
    img_show.paste(img_left, (0, 0, w, h))
    img_show.paste(img_right, (w, 0, w * 2, h))
    return np.array(img_show)
