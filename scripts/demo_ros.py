#!/usr/bin/env python

import os
import sys
import numpy as np
import open3d as o3d
import argparse
import importlib
import scipy.io as scio
from PIL import Image

import torch
from graspnetAPI import GraspGroup

import rospy
import sensor_msgs
import sensor_msgs.msg
from cv_bridge import CvBridge
import cv2
import copy

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet, pred_decode
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--num_point', type=int, default=8650, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
cfgs = parser.parse_args()

class Camera():
    def __init__(self):
        self.rgb_image = sensor_msgs.msg.Image()
        self.rgb_sub = rospy.Subscriber("/head_camera/rgb/image_raw", sensor_msgs.msg.Image, self.rgb_callback, queue_size=10)
        
        self.depth_image = sensor_msgs.msg.Image()
        self.depth_sub = rospy.Subscriber("/head_camera/depth_registered/image_raw", sensor_msgs.msg.Image, self.depth_callback, queue_size=10)
        
        self.camera_info = sensor_msgs.msg.CameraInfo()
        self.info_sub = rospy.Subscriber("/head_camera/rgb/camera_info", sensor_msgs.msg.CameraInfo, self.info_callback, queue_size=10)
            
    def rgb_callback(self, cmd_msg):
        self.rgb_image = cmd_msg
    
    def get_rgb(self):
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(self.rgb_image, desired_encoding='bgr8')
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



def get_net():
    # Init the model
    net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # Load checkpoint
    checkpoint = torch.load(cfgs.checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)"%(cfgs.checkpoint_path, start_epoch))
    # set model to eval mode
    net.eval()
    return net

def get_and_process_data(camera:Camera):
    # load data
    color = np.array(camera.get_rgb()) / 255
    depth = np.array(camera.get_depth())
    print(depth)
    # workspace_mask = np.zeros(depth.shape)
    # print(workspace_mask)
    info = camera.get_camera_info()
    intrinsic = np.array(info.K).reshape(3, 3)
    factor_depth = 0.001

    # generate cloud
    camera = CameraInfo(640.0, 480.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    # get valid points
    # mask = (workspace_mask & (depth > 0))
    # cloud_masked = cloud[mask]
    # color_masked = color[mask]
    cloud_masked = cloud
    color_masked = color
    
    # sample points
    if len(cloud_masked) >= cfgs.num_point:
        idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point-len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]
    
    # convert data
    print(cloud_masked)
    print("Data type:", cloud_masked.astype(np.float32).dtype)
    print("Data shape:", cloud_masked.shape)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(
        cloud_masked.reshape(
            cloud_masked.shape[0] * cloud_masked.shape[1], cloud_masked.shape[2]
            ).astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(
        color_masked.reshape(
            color_masked.shape[0] * color_masked.shape[1], color_masked.shape[2]
            ).astype(np.float32))
    end_points = dict()
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = cloud_sampled.to(device)
    end_points['point_clouds'] = cloud_sampled[0]
    end_points['cloud_colors'] = color_sampled

    return end_points, cloud

def get_grasps(net, end_points):
    # Forward pass
    print(end_points['point_clouds'].shape)
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    return gg

def collision_detection(gg, cloud):
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
    gg = gg[~collision_mask]
    return gg

def vis_grasps(gg, cloud):
    gg.nms()
    gg.sort_by_score()
    gg = gg[:50]
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([cloud, *grippers])

def demo(camera):
    net = get_net()
    end_points, cloud = get_and_process_data(camera)
    gg = get_grasps(net, end_points)
    if cfgs.collision_thresh > 0:
        gg = collision_detection(gg, np.array(cloud.points))
    vis_grasps(gg, cloud)

if __name__=='__main__':
    rospy.init_node("grasp_net")
    device = torch.device("cpu")
    c = Camera()
    demo(c) 
