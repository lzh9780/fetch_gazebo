#!/usr/bin/env python

# from airbot.backend import Arm, Camera, Base, Gripper
import os
import numpy as np
import copy
import rospy
# from airbot.backend.utils.utils import camera2base, armbase2world
# from airbot.lm import Detector, Segmentor
# from airbot.grasp.graspmodel import GraspPredictor
from PIL import Image
import time
import cv2
# from airbot.example.utils.draw import draw_bbox, obb2poly
# from airbot.example.utils.vis_depth import vis_image_and_depth
from scipy.spatial.transform import Rotation
from threading import Thread, Lock

def depth2cloud(depth_im, intrinsic_mat, organized=True):
    """ Generate point cloud using depth image only.
        Input:
            depth: [numpy.ndarray, (H,W), numpy.float32]
                depth image
            camera_info: dict

        Output:
            cloud: [numpy.ndarray, (H,W,3)/(H*W,3), numpy.float32]
                generated cloud, (H,W,3) for organized=True, (H*W,3) for organized=False
    """
    height, width = depth_im.shape
    fx, fy, cx, cy = intrinsic_mat[0][0], intrinsic_mat[1][1], intrinsic_mat[0][2], intrinsic_mat[1][2]
    assert (depth_im.shape[0] == height and depth_im.shape[1] == width)
    xmap = np.arange(width)
    ymap = np.arange(height)
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depth_im  # change the unit to metel
    points_x = (xmap - cx) * points_z / fx
    points_y = (ymap - cy) * points_z / fy
    cloud = np.stack([points_x, points_y, points_z], axis=-1)
    if not organized:
        cloud = cloud.reshape([-1, 3])
    return cloud

class Camera():
    def __init__(self):
        self.rgb_image = Image()
        self.rgb_sub = rospy.Subscriber("/head_camera/rgb/image_raw", Image, self.rgb_callback, queue_size=10)
        
        self.depth_image = Image()
        self.depth_sub = rospy.Subscriber("/head_camera/depth_registered/image_raw", Image, self.depth_callback, queue_size=10)
    
    def rgb_callback(self, cmd_msg):
        self.rgb_image = cmd_msg
    
    def get_rgb(self):
        return self.rgb_image
    
    def depth_callback(self, cmd_msg):
        self.depth_image = cmd_msg
    
    def get_depth(self):
        return self.depth_image

class Solution:
    def __init__(self):
        self.image_lock = Lock()
        self.result_lock = Lock()
        self.prompt_lock = Lock()
        
        self.camera = Camera()
    
    def update_once(self):
        with self.image_lock, self.result_lock:
            self._image = copy.deepcopy(self.camera.get_rgb())
            self._depth = copy.deepcopy(self.camera.get_depth())
            self._det_result = self.detector.infer(self._image, self._prompt)
            self._bbox = self._det_result['bbox'].numpy().astype(int)
            self._sam_result = self.segmentor.infer(self._image, self._bbox[None, :2][:, [1, 0]])
            self._mask = self._sam_result['mask']

    @staticmethod
    def base_cloud(image, depth, intrinsic, shift):
        cam_cloud = depth2cloud(depth, intrinsic)
        cam_cloud = np.copy(np.concatenate((cam_cloud, image), axis=2))
        return camera2base(cam_cloud, shift)

    @staticmethod
    def _vis_grasp(cloud, position, rotation):
        import open3d as o3d
        from graspnetAPI.grasp import GraspGroup
        o3d_cloud = o3d.geometry.PointCloud()
        cloud = copy.deepcopy(cloud)
        o3d_cloud.points = o3d.utility.Vector3dVector(cloud[:, :, :3].reshape(-1, 3).astype(np.float32))
        o3d_cloud.colors = o3d.utility.Vector3dVector(cloud[:, :, 3:].reshape(-1, 3).astype(np.float32) / 255.)
        gg = GraspGroup(
            np.array([1., 0.06, 0.01, 0.06, *Rotation.from_quat(rotation).as_matrix().flatten(), *position,
                      0]).reshape(1, -1))
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([o3d_cloud, *gg.to_open3d_geometry_list(), coordinate_frame])
    
    def grasp(self):
        with self.image_lock, self.result_lock:
            _depth = copy.deepcopy(self._depth)
            _image = copy.deepcopy(self._image)
            _bbox = copy.deepcopy(self._bbox)
            _mask = copy.deepcopy(self._mask)

        cloud = self.base_cloud(_image, _depth, self.camera.INTRINSIC, self.CAMERA_SHIFT)

        grasp_position = cloud[ _bbox[0], _bbox[1] - _bbox[3] // 2 + 8][:3]
        grasp_position[2] = -0.168
        grasp_rotation = Rotation.from_euler('xyz', [0, np.pi / 2, 0], degrees=False).as_quat()

        self.arm.move_end_to_pose(grasp_position, grasp_rotation)