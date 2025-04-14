import os
import json
import numpy as np
import pyrealsense2 as rs
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from pycocotools import mask as maskUtils
from .mrcnn import model as modellib, visualize
from .mrcnn.config import Config

# Mask R-CNN Inference 설정
class InferenceConfig(Config):
    NAME = "object"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.85
    NUM_CLASSES = 1 + 3  # 배경 + (tanger, yeolgwa, godoo)

inference_config = InferenceConfig()

# 정확한 모델 파일 경로
MODEL_PATH = "/root/mask_rcnn_ros2_ws/src/mask_rcnn_ros2/mask_rcnn_ros2/logs/custom20250413T0821/mask_rcnn_custom_0015.h5"
MODEL_DIR = "/root/mask_rcnn_ros2_ws/src/mask_rcnn_ros2/mask_rcnn_ros2/logs"

# Mask R-CNN 모델 로딩
test_model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=MODEL_DIR)
test_model.load_weights(MODEL_PATH, by_name=True)

class MaskRCNNNode(Node):
    def __init__(self):
        super().__init__('mask_rcnn_node')

        self.publisher_ = self.create_publisher(String, '/mask/result', 10)
        self.image_publisher = self.create_publisher(Image, '/mask/image_result', 10)
        self.original_image_publisher = self.create_publisher(Image, '/mask/original_image', 10)
        self.bridge = CvBridge()

        # Realsense 초기화
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)

        self.timer = self.create_timer(1.0 / 30.0, self.process_frame)
        self.image_id = 1
        self.annotation_id = 1

    def process_frame(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return

        color_image_bgr = np.asanyarray(color_frame.get_data())

        # 퍼블리시: 원본
        try:
            original_image_msg = self.bridge.cv2_to_imgmsg(color_image_bgr, encoding="bgr8")
            self.original_image_publisher.publish(original_image_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish original image: {e}")
            return

        # 추론
        rgb_image = color_image_bgr[:, :, ::-1]  # BGR to RGB
        results = test_model.detect([rgb_image], verbose=0)
        r = results[0]

        # # 시각화
        # display_img_rgb = visualize.display_instances(
        #     rgb_image, r['rois'], r['masks'], r['class_ids'],
        #     ["BG", "tanger", "yeolgwa", "godoo"], r['scores'],
        #     show_mask=True, show_bbox=True
        # )

        # if display_img_rgb is not None:
        #     try:
        #         display_img_bgr = display_img_rgb[:, :, ::-1]  # RGB to BGR
        #         image_msg = self.bridge.cv2_to_imgmsg(display_img_bgr, encoding="bgr8")
        #         self.image_publisher.publish(image_msg)
        #     except Exception as e:
        #         self.get_logger().error(f"Failed to publish result image: {e}")

        try:
            mask_data = self.format_json_result(r)
            msg = String()
            msg.data = json.dumps(mask_data)
            self.publisher_.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish JSON result: {e}")

        self.image_id += 1

    def format_json_result(self, result):
        coco_output = {
            "info": {"description": "ROS2 Mask R-CNN Inference"},
            "images": [],
            "annotations": [],
            "categories": [
                {"id": 1, "name": "tanger"},
                {"id": 2, "name": "yeolgwa"},
                {"id": 3, "name": "godoo"}
            ]
        }

        img_info = {
            "id": self.image_id,
            "width": 640,
            "height": 480,
            "file_name": f"frame_{self.image_id}.jpg"
        }
        coco_output["images"].append(img_info)

        for i in range(len(result["class_ids"])):
            class_id = int(result["class_ids"][i])
            score = float(result["scores"][i])
            y1, x1, y2, x2 = result["rois"][i]
            bbox = [x1, y1, x2 - x1, y2 - y1]

            mask = result["masks"][:, :, i]
            rle = maskUtils.encode(np.asfortranarray(mask))
            rle["counts"] = rle["counts"].decode("utf-8")

            annotation = {
                "id": self.annotation_id,
                "image_id": self.image_id,
                "category_id": class_id,
                "segmentation": rle,
                "area": int(np.sum(mask)),
                "bbox": bbox,
                "iscrowd": 0,
                "score": score
            }
            coco_output["annotations"].append(annotation)
            self.annotation_id += 1

        return coco_output

    def shutdown(self):
        self.pipeline.stop()


def main(args=None):
    rclpy.init(args=args)
    node = MaskRCNNNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.shutdown()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()