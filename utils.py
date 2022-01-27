import argparse
import onnx
from transforms import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--det_model_dir",type=str, help="Path to onnx model weigts")
    parser.add_argument("-i", "--image_path",type=str, help="Path to test image")
    parser.add_argument("--onnx_check", action="store_true", help="Pass if want to check onnx model with onnx.checker")
    # DB params
    parser.add_argument("--det_db_thresh", type=float, default=0.3)
    parser.add_argument("--det_db_box_thresh", type=float, default=0.6)
    parser.add_argument("--det_db_unclip_ratio", type=float, default=1.5)
    parser.add_argument("--max_batch_size", type=int, default=10)
    parser.add_argument("--use_dilation", type=bool, default=False)
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
    assert isinstance(op_param_list, list), ('operator config should be a list')
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