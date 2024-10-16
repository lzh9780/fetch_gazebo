# Usage of fetch_gazebo

Based on https://github.com/ZebraDevs/fetch_gazebo/tree/gazebo11

## 1. Requests
`sudo apt-get install ros-noetic-fetch-moveit-config`

## 2. Start Gazebo
`roscore` Optional, Recommanded

`roslaunch fetch_gazebo simple_grasp.launch`

## 3. Start Moveit
Ik could be closed. Change the `use_ik` at line 21 to `False` could disable ik. If disabled, this step could be skipped. 

If the ik is not disabled:

`rosalunch fetch_moveit_config move_group.launch` 

## 4. Start control scripts
`rosrun fetch_gazebo interface.py`

## 5. Control topics
### Camera (head) controllor:
```
rostopic pub -1 /fetch/camera_orientation geometry_msgs/Point "x: 0.0
y: 0.0
z: 0.0"
```

Direction: 

positive x: up

negative x: down 

positive z: left

negative z: right

### Arm controllor (with ik)
```
rostopic pub -1 /fetch/pose_cmd geometry_msgs/PoseStamped "header:
  seq: 0
  stamp:
    secs: 0
    nsecs: 0
  frame_id: ''
pose:
  position:
    x: 0.0
    y: 0.0
    z: 0.0
  orientation:
    x: 0.0
    y: 0.0
    z: 0.0
    w: 0.0"  
```


