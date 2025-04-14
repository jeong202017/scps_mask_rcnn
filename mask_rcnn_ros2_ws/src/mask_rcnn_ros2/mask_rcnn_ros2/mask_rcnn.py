import os
import numpy as np
import pyrealsense2 as rs
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from .mrcnn import model as modellib
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

        self.publisher_ = self.create_publisher(Float64MultiArray, '/mask/statistics', 10)
        self.image_publisher = self.create_publisher(Image, '/mask/image_result', 10)
        self.original_image_publisher = self.create_publisher(Image, '/mask/original_image', 10)
        self.bridge = CvBridge()

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipeline.start(config)

        self.timer = self.create_timer(1.0 / 30.0, self.process_frame)
        self.image_id = 1
        self.annotation_id = 1

        self.last_update_time = self.get_clock().now()
        self.stats = {1: [], 2: [], 3: []}  # {class_id: [diameters]}

        # Get camera intrinsics
        profile = self.pipeline.get_active_profile()
        depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        self.intrinsics = depth_stream.get_intrinsics()  # rs.intrinsics

    def process_frame(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return

        color_image_bgr = np.asanyarray(color_frame.get_data())

        try:
            original_image_msg = self.bridge.cv2_to_imgmsg(color_image_bgr, encoding="bgr8")
            self.original_image_publisher.publish(original_image_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish original image: {e}")
            return

        rgb_image = color_image_bgr[:, :, ::-1]  # BGR to RGB
        results = test_model.detect([rgb_image], verbose=0)
        r = results[0]

        for i in range(len(r["class_ids"])):
            class_id = int(r["class_ids"][i])
            if class_id not in [1, 2, 3]:
                continue

            mask = r["masks"][:, :, i].astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            max_diameter_cm = 0
            for cnt in contours:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                x, y, radius = int(x), int(y), int(radius)
                diameter_px = radius * 2

                # Depth at center point (in meters)
                depth = depth_frame.get_distance(x, y)
                if depth == 0:
                    continue

                # Convert pixel diameter to cm using fx
                fx = self.intrinsics.fx
                diameter_m = (diameter_px * depth) / fx  # width in meters
                diameter_cm = diameter_m * 100

                if diameter_cm > max_diameter_cm:
                    max_diameter_cm = diameter_cm

            self.stats[class_id].append(max_diameter_cm)

        now = self.get_clock().now()
        if (now - self.last_update_time).nanoseconds / 1e9 >= 2.0:
            for class_id in [1, 2, 3]:
                count = len(self.stats[class_id])
                max_diameter = max(self.stats[class_id]) if self.stats[class_id] else 0.0
                msg = Float64MultiArray()
                msg.data = [float(class_id), float(count), float(max_diameter)]
                self.publisher_.publish(msg)
                self.get_logger().info(f"Published stats - class {class_id}: count={count}, max_diameter={max_diameter:.2f} cm")
            self.stats = {1: [], 2: [], 3: []}
            self.last_update_time = now

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
