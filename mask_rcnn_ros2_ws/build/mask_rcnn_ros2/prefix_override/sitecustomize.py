import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/root/mask_rcnn_ros2_ws/install/mask_rcnn_ros2'
