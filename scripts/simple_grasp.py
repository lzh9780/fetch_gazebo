import rospy
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from control_msgs.msg import GripperCommandActionGoal
from interface import SimpleGraspInterface
import time

def main():
    grasp = SimpleGraspInterface()
    
    while not rospy.is_shutdown():
        grasp.gripper_control()
    

if __name__ == "__main__":
    main()