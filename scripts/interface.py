#!/usr/bin/env python

import rospy
from gazebo_msgs.msg import LinkState, ModelState
from std_msgs.msg import Bool
from moveit_python import MoveGroupInterface, PlanningSceneInterface
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from control_msgs.msg import GripperCommandActionGoal
import tf
import tf.transformations
import time

class SimpleGraspInterface:
    def __init__(self, node_name: str,):
        self.node_name = node_name
        rospy.init_node(self.node_name)
        
        self.move_group = MoveGroupInterface("arm_with_torso", "base_link")
        print("movegroup loaded")
        
        self.planning_scene = PlanningSceneInterface("base_link")
        # self.planning_scene.removeCollisionObject("my_front_ground")
        # self.planning_scene.removeCollisionObject("my_back_ground")
        # self.planning_scene.removeCollisionObject("my_right_ground")
        # self.planning_scene.removeCollisionObject("my_left_ground")
        self.planning_scene.addCube("my_front_ground", 2, 1.1, 0.0, -1.0)
        self.planning_scene.addCube("my_back_ground", 2, -1.2, 0.0, -1.0)
        self.planning_scene.addCube("my_left_ground", 2, 0.0, 1.2, -1.0)
        self.planning_scene.addCube("my_right_ground", 2, 0.0, -1.2, -1.0)
        print("Loaded")

        # This is the wrist link not the gripper itself
        self.gripper_frame = 'gripper_link'
        
        self.gripper_pose_stamped = PoseStamped()
        self.gripper_pose_stamped.header.frame_id = 'base_link'
        
        self.sub_queue_size = 10
        self.pub_queue_size = 10
        
        self.start_time = rospy.Time.now()
        self.curr_ee_pose = Pose
        self.target_pose = Pose
        
        self.curr_link_pose_sub = rospy.Subscriber(
            "/gazebo/link_states", LinkState, self.link_callback, queue_size=self.sub_queue_size
        )
        
        self.reset_sub = rospy.Subscriber(
            "/reset", Bool, self.reset_callback, queue_size=self.sub_queue_size
        )
        
        self.model_sub = rospy.Subscriber(
            "/gazebo/model_states", ModelState, self.model_callback, queue_size=self.sub_queue_size
        )
        
        self.gripper_pub = rospy.Publisher(
            "/gripper_controller/gripper_action/goal", GripperCommandActionGoal, queue_size=self.pub_queue_size
        )
    
    def arrived(self, target_pose:Pose):
        if self.curr_ee_pose.position == target_pose.position \
            and self.curr_ee_pose.orientation == target_pose.orientation:
            return True
        else:
            return False
    
    def arm_movement(self, pose:Pose):
        self.gripper_pose_stamped.pose = pose
        self.gripper_pose_stamped.header.stamp = rospy.Time.now()
        self.move_group.moveToPose(self.gripper_pose_stamped, self.gripper_frame)
        while not self.arrived(pose):
            continue
    
    def link_callback(self, cmd_msg:LinkState):
        link_name = cmd_msg.link_name
        link_pose = cmd_msg.pose
        c = 0
        while link_name[c] != 'gripper_link':
            c += 1
        self.curr_ee_pose = link_pose[c]
    
    def model_callback(self, cmd_msg:ModelState):
        c = 0
        while cmd_msg.model_name[c] != "demo_cube":
            c += 1
        pose:Pose = cmd_msg.pose[c]
        pose.position.x -= 0.05
        self.target_pose = pose
    
    def gripper_control(self, cmd:int):
        """
        Args:
            cmd (int): Enter 1 to open the gripper, -1 to close the gripper
        """
        cmd_msg = GripperCommandActionGoal
        cmd_msg.goal.command.max_effort = 10.0
        cmd_msg.goal.command.position = 0.1 * cmd
        self.gripper_pub.publish(cmd_msg)
    
    def reset_callback(self, cmd_msg:Bool):
        if cmd_msg.data:
            pass
    
    def run(self):
        """
        step 1: move arm to the target
        step 2: close gripper
        step 3: move arm to other position
        """
        step = 1
        while not rospy.is_shutdown:
            # if rospy.Time.now() - self.start_time > 5:
            ee_pos:Point = self.curr_ee_pose.position
            ee_ori:Quaternion = self.curr_ee_pose.orientation
            ee_rot = tf.transformations.euler_from_quaternion(ee_ori)
            if step == 1:
                self.arm_movement(self.target_pose)
                step += 1
            elif step == 2:
                self.gripper_control(-1)
                time.sleep(3)
                step += 1
            elif step == 3:
                self.target_pose.position.z += 0.2
                self.arm_movement(self.target_pose)
            print(step)
            continue
    

def main():
    g = SimpleGraspInterface("simple_grasp")
    g.run()

if __name__ == '__main__':
    main()
