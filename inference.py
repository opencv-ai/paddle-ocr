from pipeline import TextPipeline
from detector import TextDetector
from utils import parse_args, model_check, draw_text_boxes, draw_ocr_box_txt
import cv2
import os
from PIL import Image
from tqdm import tqdm
import numpy as np
import json


def inference(args):
    if args.onnx_check:
        model_check(args.det_model_dir)
        model_check(args.rec_model_dir)
    if os.path.isdir(args.image_path):
        image_paths = [os.path.join(args.image_path, img) for img in os.listdir(args.image_path)]
    else:
        image_paths = [args.image_path]
    text_detector = TextDetector(args)
    for image_path in image_paths:
        print("Processing: ", image_path)
        input_image = cv2.imread(image_path)
        if args.run_detection:
            # Inference
            boxes, _ = text_detector(input_image)
            print(f"{boxes.shape[0]} boxes were found")
            # Visualization
            vis_image = draw_text_boxes(boxes, input_image.copy())
            result_vis_image_name = os.path.basename(image_path)
            cv2.imwrite(f"images/results/{result_vis_image_name}", vis_image)
        elif args.run_pipeline:
            text_sys = TextPipeline(args)
            font_path = args.vis_font_path
            drop_score = args.drop_score
            dt_boxes, rec_res = text_sys(input_image)
            image = Image.fromarray(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
            boxes = dt_boxes
            txts = [rec_res[i][0] for i in range(len(rec_res))]
            scores = [rec_res[i][1] for i in range(len(rec_res))]

            draw_img = draw_ocr_box_txt(
                image,
                boxes,
                txts,
                scores,
                drop_score=drop_score,
                font_path=font_path)
            result_vis_image_name = os.path.basename(image_path)
            cv2.imwrite(f"images/results/{result_vis_image_name}", cv2.cvtColor(draw_img, cv2.COLOR_RGB2BGR))
        else:
            raise NotImplementedError("We support either text detection or full pipeline")


if __name__ == "__main__":
    inference(parse_args())

