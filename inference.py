from detector import TextDetector
from utils import parse_args, model_check, draw_text_boxes
import cv2
import os

def inference(args):
    if args.onnx_check:
        model_check(args.det_model_dir)

    input_image = cv2.imread(args.image_path)
    
    # Inference
    text_detector = TextDetector(args)
    boxes, _ = text_detector(input_image)
    print(f"{boxes.shape[0]} boxes were found")
    # Visualization
    vis_image = draw_text_boxes(boxes, input_image.copy())
    result_vis_image_name = os.path.basename(args.image_path)
    cv2.imwrite(f"images/results/{result_vis_image_name}", vis_image)

if __name__ == "__main__":
    args = parse_args()
    inference(args)