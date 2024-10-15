# Usage of fetch_gazebo

Based on https://github.com/ZebraDevs/fetch_gazebo/tree/gazebo11

## 1. Requests
`sudo apt-get install ros-noetic-fetch-moveit-config`

## 2. Start Gazebo
`roslaunch fetch_gazebo simple_grasp.launch`

## 3. Start Moveit
`rosalunch fetch_moveit_config move_group.launch`

Topics:

`/gazebo/set_model_state`: change model pose

`/gazebo/model_states`: get model pose

`/gazebo/link_states`: get link pose

## 4. Start scripts
`rosrun fetch_gazebo script_name.py`

Currently only `disco.py` can be used# fetch_gazebo
