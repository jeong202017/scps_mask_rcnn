import os
import json
import cv2
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

# Mask R-CNN 모델 로드
# MODEL_DIR = "/root/mask_rcnn_ros2_ws/src/mask_rcnn_ros2/mask_rcnn_ros2/logs/custom20250413T0821/mask_rcnn_custom_0015.h5"
# test_model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=MODEL_DIR)
# model_path = test_model.find_last()
# test_model.load_weights(model_path, by_name=True)

# 정확한 모델 파일 경로 (.h5 파일)
MODEL_PATH = "/root/mask_rcnn_ros2_ws/src/mask_rcnn_ros2/mask_rcnn_ros2/logs/custom20250413T0821/mask_rcnn_custom_0015.h5"

# model_dir은 logs 디렉토리까지만 지정
MODEL_DIR = "/root/mask_rcnn_ros2_ws/src/mask_rcnn_ros2/mask_rcnn_ros2/logs"

# Mask R-CNN 객체 생성
test_model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=MODEL_DIR)

# 명시적으로 weight 로드
test_model.load_weights(MODEL_PATH, by_name=True)



# ROS2 노드 클래스 정의
class MaskRCNNNode(Node):
    def __init__(self):
        super().__init__('mask_rcnn_node')
        
        # ROS2 퍼블리셔
        self.publisher_ = self.create_publisher(String, '/mask/result', 10)
        self.image_publisher = self.create_publisher(Image, '/mask/image_result', 10)

        # CvBridge 객체 생성
        self.bridge = CvBridge()

        # Realsense D435i 초기화
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)

        # 타이머 생성 (30Hz)
        self.timer = self.create_timer(1.0 / 30.0, self.process_frame)

        self.image_id = 1
        self.annotation_id = 1

    def process_frame(self):
        # Realsense에서 프레임 가져오기
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return
        
        # OpenCV 형식으로 변환
        color_image = np.asanyarray(color_frame.get_data())

        # Mask R-CNN Inference 수행
        results = test_model.detect([color_image], verbose=0)
        r = results[0]

        # 시각화
        display_img = visualize.display_instances(color_image, r['rois'], r['masks'], r['class_ids'],
                                                  ["BG", "tanger", "yeolgwa", "godoo"], r['scores'],
                                                  show_mask=True, show_bbox=True)
        
        cv2.imshow("Instance Segmentation Result", display_img)
        cv2.waitKey(1)

        # ROS2 토픽 퍼블리시
        mask_data = self.format_json_result(r)
        msg = String()
        msg.data = json.dumps(mask_data)
        self.publisher_.publish(msg)

        # 결과 이미지를 ROS2 토픽으로 퍼블리시
        image_msg = self.bridge.cv2_to_imgmsg(display_img, encoding="bgr8")
        self.image_publisher.publish(image_msg)

        self.image_id += 1

    def format_json_result(self, result):
        """ Mask R-CNN 결과를 JSON 형식으로 변환 """
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

            # Bounding Box 변환
            y1, x1, y2, x2 = result["rois"][i]
            bbox = [x1, y1, x2 - x1, y2 - y1]

            # Segmentation Mask 변환
            mask = result["masks"][:, :, i]
            rle = maskUtils.encode(np.asfortranarray(mask))
            rle["counts"] = rle["counts"].decode("utf-8")  # JSON 직렬화

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
        """ 노드 종료 시 Realsense 정리 """
        self.pipeline.stop()
        cv2.destroyAllWindows()

# ROS2 실행
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
