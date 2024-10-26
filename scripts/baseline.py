#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool
from sensor_msgs.msg import Image, CameraInfo, JointState
from geometry_msgs.msg import PoseStamped, Point
from control_msgs.msg import GripperCommandActionGoal
from cv_bridge import CvBridge
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import torch
import tf2_ros
from scipy.spatial.transform import Rotation as R

class Camera():
    def __init__(self):
        self.rgb_image = Image()
        self.rgb_sub = rospy.Subscriber("/head_camera/rgb/image_raw", Image, self.rgb_callback, queue_size=10)
        
        self.depth_image = Image()
        self.depth_sub = rospy.Subscriber("/head_camera/depth_registered/image_raw", Image, self.depth_callback, queue_size=10)
        
        self.camera_info = CameraInfo()
        self.info_sub = rospy.Subscriber("/head_camera/rgb/camera_info", CameraInfo, self.info_callback, queue_size=10)
            
    def rgb_callback(self, cmd_msg):
        self.rgb_image = cmd_msg
    
    def get_rgb(self):
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(self.rgb_image, desired_encoding='rgb8')
        return cv_image
    
    def depth_callback(self, cmd_msg):
        self.depth_image = cmd_msg
    
    def get_depth(self):
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(self.depth_image, desired_encoding='32FC1')
        return cv_image
    
    def info_callback(self, cmd_msg):
        self.camera_info = cmd_msg
        
    def get_camera_info(self):
        return self.camera_info

class Solver:
    def __init__(self):
        self.camera = Camera()
        self.model = self.load_model()
        self.classes = self.model.names
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.img_pub = rospy.Publisher("/detect/image", Image, queue_size=10)
        
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        
        self.arm_pub = rospy.Publisher("/fetch/pose_cmd", PoseStamped, queue_size=10)
        self.gripper_pub = rospy.Publisher(
            "/fetch/gripper_cmd", Bool, queue_size=10
        )
        self.joint_pub = rospy.Publisher("/fetch/joint_cmd", JointState, queue_size=10)
        self.head_pub = rospy.Publisher("/fetch/camera_orientation", Point, queue_size=10)
    
    def load_model(self):
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)
        return model

    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels = results.xyxyn[0][:, -1].cpu().numpy()
        cord = results.xyxyn[0][:, :-1].cpu().numpy()
        return labels, cord

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    
    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
                print(labels[i], self.class_to_label(labels[i]))

        return frame
    
    def detect(self):
        time.sleep(5)
        # rgb = self.camera.get_rgb()
        # depth = self.camera.get_depth()
        
        frame = self.camera.get_rgb()
        results = self.score_frame(frame)
        print(results)
        print("Result get")
        frame_box = self.plot_boxes(results, frame)
        
        labels, cord = results
        for i in range(len(labels)):
            if labels[i] == 60:
                continue
            else:
                row = cord[i]
                x_shape, y_shape = frame.shape[1], frame.shape[0]
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                x = (x1+x2) / 2
                y = (y1+y2) / 2
                print(x, y)
                return x, y
        
        return -1, -1

    def tf_cal(self):
        # Wait for the transform to be available
        try:
            trans = self.tfBuffer.lookup_transform('base_link', 'head_camera_rgb_optical_frame', rospy.Time.now(), rospy.Duration(1.0))
            return trans
            # trans.transform contains the transformation matrix
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logerr("Transform not available")

        
if __name__ == "__main__":
    rospy.init_node("grasp")
    solver = Solver()
    
    # point = Point()
    # point.x = -0.6
    # point.y = 0
    # point.z = 0
    # solver.head_pub.publish(point)
    # time.sleep(2)
       
    cmd_msg = Bool()
    cmd_msg.data = False
    solver.gripper_pub.publish(cmd_msg)
    
    x, y = -1, -1
    
    while x == -1 and y == -1:
        x, y = solver.detect()
    
    print("x, y,", x, y)
    
    info = solver.camera.get_camera_info()
    k = np.array(info.K).reshape(3, 3)
    fx = k[0][0]
    cx = k[0][2]
    fy = k[1][1]
    cy = k[1][2]
    print(k)
    depth = solver.camera.get_depth()
    d = np.array(depth)[int(y)][int(x)]
    pos_x = (x - cx) * d / fx
    pos_y = (y - cy) * d / fy
    pos_z = d
    
    print(pos_x, pos_y, pos_z)
    pos = [pos_x, pos_y, pos_z, 1]
    
    trans = solver.tf_cal().transform
    print(trans.translation, trans.rotation)
    translation = [trans.translation.x, trans.translation.y, trans.translation.z]
    
    rotation = R.from_quat([trans.rotation.x, trans.rotation.y, trans.rotation.z, trans.rotation.w]).as_matrix()
    
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation
    transformation_matrix[:3, 3] = translation
    
    print(transformation_matrix)
    
    transformed_position = (transformation_matrix @ pos)[:3]
    print(transformed_position)
    
    pose = PoseStamped()
    pose.pose.position.x = transformed_position[0] - 0.1
    pose.pose.position.y = transformed_position[1]
    pose.pose.position.z = transformed_position[2] + 0.1
    pose.pose.orientation.x = 0
    pose.pose.orientation.y = 0
    pose.pose.orientation.z = 0
    pose.pose.orientation.w = 1
    
    solver.arm_pub.publish(pose)
    
    time.sleep(5)
    
    pose.pose.position.x += 0.15
    solver.arm_pub.publish(pose)
    
    print(pose)
    
    time.sleep(5)
    
    """
    Args:
        cmd (int): Enter 1 to open the gripper, -1 to close the gripper
    """
    cmd_msg = Bool()
    cmd_msg.data = True
    solver.gripper_pub.publish(cmd_msg)
    
    time.sleep(3)
    
    joint = JointState()
    joint.position = [1.32, 0.7, 0.0, -2.0, 0.0, -0.57, 0.0]

    solver.joint_pub.publish(joint)
    
    print(joint.position)
    