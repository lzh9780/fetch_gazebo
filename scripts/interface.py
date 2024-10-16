#!/usr/bin/env python

import argparse
import rospy
from threading import Thread, Event
from gazebo_msgs.msg import LinkState, ModelState
from std_msgs.msg import Bool
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest, GetPositionIKResponse
from moveit_python import MoveGroupInterface, PlanningSceneInterface
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from control_msgs.msg import GripperCommandActionGoal, FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import tf
import tf.transformations
import time
import numpy as np
from scipy.spatial.transform import Rotation
import actionlib

class SimpleGraspInterface:
    def __init__(self, node_name: str, use_ik=True):
        self.node_name = node_name
        rospy.init_node(self.node_name)
        
        rospy.loginfo("Waiting for head_controller...")
        self.head_client = actionlib.SimpleActionClient("head_controller/follow_joint_trajectory", FollowJointTrajectoryAction)
        self.head_client.wait_for_server()
        rospy.loginfo("...connected.")

        rospy.loginfo("Waiting for arm_controller...")
        self.arm_client = actionlib.SimpleActionClient("arm_controller/follow_joint_trajectory", FollowJointTrajectoryAction)
        self.arm_client.wait_for_server()
        rospy.loginfo("...connected.")
        
        # self.move_group = MoveGroupInterface("arm_with_torso", "base_link")
        
        # self.planning_scene = PlanningSceneInterface("base_link")
        # # self.planning_scene.removeCollisionObject("my_front_ground")
        # # self.planning_scene.removeCollisionObject("my_back_ground")
        # # self.planning_scene.removeCollisionObject("my_right_ground")
        # # self.planning_scene.removeCollisionObject("my_left_ground")
        # self.planning_scene.addCube("my_front_ground", 2, 1.1, 0.0, -1.0)
        # self.planning_scene.addCube("my_back_ground", 2, -1.2, 0.0, -1.0)
        # self.planning_scene.addCube("my_left_ground", 2, 0.0, 1.2, -1.0)
        # self.planning_scene.addCube("my_right_ground", 2, 0.0, -1.2, -1.0)
        
        self.arm_joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "upperarm_roll_joint",
              "elbow_flex_joint", "forearm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"]
        self.head_joint_names = ["head_pan_joint", "head_tilt_joint"]
        
        self._moveit_ik_service = rospy.ServiceProxy('/compute_ik', GetPositionIK)
        self.ik_event = Event()
        
        self.ik_target = PoseStamped()
        
        print(use_ik)
        if use_ik:
            self.arm_sub = rospy.Subscriber("/fetch/pose_cmd", PoseStamped, self.arm_ik_callback, queue_size=10)
            self.ik_target = None
            def compute_ik():
                while not rospy.is_shutdown():
                    self.ik_event.wait()
                    joint_target = self._compute_inverse_kinematics(self.ik_target)
                    if joint_target is not None:
                        self.arm_joint_pos_target = joint_target
                        trajectory = JointTrajectory()
                        trajectory.joint_names = self.arm_joint_names
                        trajectory.points.append(JointTrajectoryPoint())
                        trajectory.points[0].positions = joint_target
                        trajectory.points[0].velocities =  [0.0] * len(joint_target)
                        trajectory.points[0].accelerations = [0.0] * len(joint_target)
                        trajectory.points[0].time_from_start = rospy.Duration(4.0)
                        
                        arm_goal = FollowJointTrajectoryGoal()
                        arm_goal.trajectory = trajectory
                        arm_goal.goal_time_tolerance = rospy.Duration(0.0)
                        
                        self.arm_client.send_goal(arm_goal)
                        
                    self.ik_event.clear()

            Thread(target=compute_ik, daemon=True).start()

        # This is the wrist link not the gripper itself
        self.gripper_frame = 'gripper_link'
        
        self.gripper_pose_stamped = PoseStamped()
        self.gripper_pose_stamped.header.frame_id = 'base_link'
        
        self.arm_joint_pos_target = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.sub_queue_size = 10
        self.pub_queue_size = 10
        
        self.start_time = rospy.Time.now()
        self.curr_ee_pose = Pose
        self.target_pose = Pose
        
        # self.curr_link_pose_sub = rospy.Subscriber(
        #     "/gazebo/link_states", LinkState, self.link_callback, queue_size=self.sub_queue_size
        # )
        
        # self.reset_sub = rospy.Subscriber(
        #     "/reset", Bool, self.reset_callback, queue_size=self.sub_queue_size
        # )
        
        # self.model_sub = rospy.Subscriber(
        #     "/gazebo/model_states", ModelState, self.model_callback, queue_size=self.sub_queue_size
        # )
        
        self.gripper_pub = rospy.Publisher(
            "/gripper_controller/gripper_action/goal", GripperCommandActionGoal, queue_size=self.pub_queue_size
        )
        
        self.head_sub = rospy.Subscriber(
            "/fetch/camera_orientation", Point, self.head_callback, queue_size=self.sub_queue_size
        )
        
    def _get_joint_states_by_names(self, joint_names:tuple,all_joint_names:tuple,all_joint_values:tuple) -> list:
        joint_states = [0 for _ in range(len(joint_names))]
        for i, name in enumerate(joint_names):
            joint_states[i] = all_joint_values[all_joint_names.index(name)]
        return joint_states
    
    def _compute_inverse_kinematics(self, target_pose:PoseStamped):
        ik_request = GetPositionIKRequest()
        ik_request.ik_request.group_name = "arm_with_torso"
        ik_request.ik_request.ik_link_name = "gripper_link"
        ik_request.ik_request.pose_stamped = target_pose
        ik_request.ik_request.pose_stamped.header.frame_id = "base_link"
        ik_request.ik_request.avoid_collisions = True
        ik_request.ik_request.timeout = rospy.Duration(1)
        ik_response:GetPositionIKResponse = self._moveit_ik_service.call(ik_request)
        if ik_response.error_code.val == ik_response.error_code.SUCCESS:
            target_joint_states = self._get_joint_states_by_names(self.arm_joint_names,
                                            ik_response.solution.joint_state.name, ik_response.solution.joint_state.position)
            return target_joint_states
        else:
            rospy.logerr(f"IK failed with error code: {ik_response.error_code.val}")
            return None
    
    # def arrived(self, target_pose:Pose):
    #     if self.curr_ee_pose.position == target_pose.position \
    #         and self.curr_ee_pose.orientation == target_pose.orientation:
    #         return True
    #     else:
    #         return False
    
    # def arm_movement(self, pose:Pose):
    #     self.gripper_pose_stamped.pose = pose
    #     self.gripper_pose_stamped.header.stamp = rospy.Time.now()
    #     self.move_group.moveToPose(self.gripper_pose_stamped, self.gripper_frame)
    #     while not self.arrived(pose):
    #         continue
    
    # def link_callback(self, cmd_msg:LinkState):
    #     link_name = cmd_msg.link_name
    #     link_pose = cmd_msg.pose
    #     c = 0
    #     while link_name[c] != 'gripper_link':
    #         c += 1
    #     self.curr_ee_pose = link_pose[c]
    
    # def model_callback(self, cmd_msg:ModelState):
    #     c = 0
    #     while cmd_msg.model_name[c] != "demo_cube":
    #         c += 1
    #     pose:Pose = cmd_msg.pose[c]
    #     pose.position.x -= 0.05
    #     self.target_pose = pose
    
    def arm_ik_callback(self, cmd_msg):
        self.ee_target_position = np.array([
            cmd_msg.pose.position.x, 
            cmd_msg.pose.position.y, 
            cmd_msg.pose.position.z
        ])
        self.ee_target_rotation = Rotation.from_quat(np.array(
            [
                cmd_msg.pose.orientation.x,
                cmd_msg.pose.orientation.y,
                cmd_msg.pose.orientation.z,
                cmd_msg.pose.orientation.w,
            ]
        ))

        self.ik_target = cmd_msg
        self.ik_event.set()
    
    def gripper_control(self, cmd:int):
        """
        Args:
            cmd (int): Enter 1 to open the gripper, -1 to close the gripper
        """
        cmd_msg = GripperCommandActionGoal
        cmd_msg.goal.command.max_effort = 10.0
        cmd_msg.goal.command.position = 0.1 * cmd
        self.gripper_pub.publish(cmd_msg)
    
    def head_callback(self, cmd_msg:Point):
        x, y, z = cmd_msg.x, cmd_msg.y, cmd_msg.z
        trajectory = JointTrajectory()
        trajectory.joint_names = self.head_joint_names
        trajectory.points.append(JointTrajectoryPoint())
        trajectory.points[0].positions = [z, -x]
        trajectory.points[0].velocities = [0.0] * len(self.head_joint_names)
        trajectory.points[0].accelerations = [0.0] * len(self.head_joint_names)
        trajectory.points[0].time_from_start = rospy.Duration(5.0)

        head_goal = FollowJointTrajectoryGoal()
        head_goal.trajectory = trajectory
        head_goal.goal_time_tolerance = rospy.Duration(0.0)
        
        self.head_client.send_goal(head_goal)
    
    def reset_callback(self, cmd_msg:Bool):
        if cmd_msg.data:
            pass
    
    def run(self):
        """
        step 1: move arm to the target
        step 2: close gripper
        step 3: move arm to other position
        """
        # step = 1
        # while not rospy.is_shutdown:
        #     # if rospy.Time.now() - self.start_time > 5:
        #     ee_pos:Point = self.curr_ee_pose.position
        #     ee_ori:Quaternion = self.curr_ee_pose.orientation
        #     ee_rot = tf.transformations.euler_from_quaternion(ee_ori)
        #     if step == 1:
        #         self.arm_movement(self.target_pose)
        #         print("Planing")
        #         step += 1
        #     elif step == 2:
        #         self.gripper_control(-1)
        #         time.sleep(3)
        #         step += 1
        #     elif step == 3:
        #         self.target_pose.position.z += 0.2
        #         self.arm_movement(self.target_pose)
        #     print(step)
        #     continue
        rospy.spin()
    

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--ik', action='store_true')
    # args, unknown = parser.parse_known_args()

    # g = SimpleGraspInterface("simple_grasp", use_ik=args.ik)
    g = SimpleGraspInterface("simple_grasp")
    g.run()

if __name__ == '__main__':
    main()
