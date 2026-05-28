# ARX Real Robot Workspace Calibration

Fill in these values by pushing the robot arm to each extreme with the
teach pendant and recording the EEF coordinates from the controller.

## Workspace bounds (robot base frame, meters)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `x_min` | TBD | Furthest back reach |
| `x_max` | TBD | Furthest forward reach |
| `y_min` | TBD | Furthest left reach |
| `y_max` | TBD | Furthest right reach |
| `z_min` | TBD | Table surface height |
| `z_max` | TBD | Highest safe reach |

## Derived parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `pre_pos_z` | `z_min + 0.20` | Safe approach height above table |
| `carry_z` | `z_min + 0.25` | Safe transport height |
| `release_z` | `z_min + 0.05` | Release height above surface |

## Gripper calibration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Gripper fully open | TBD | Read from controller display |
| Gripper fully closed | TBD | Read from controller display |
| gripper_closed_thresh | TBD | Gripper value below which object is grasped |

## Object offset calibration

For each object type, perform one grasp, measure the offset between
the object's 3D perception position and the actual grasp position.

| Object | EEF-object offset (x,y,z) | Notes |
|--------|---------------------------|-------|
| TBD | TBD | |

## Camera mapping

| openpi slot | Physical camera | Notes |
|-------------|-----------------|-------|
| `left_wrist_view` | TBD | |
| `face_view` | TBD | Front-facing camera |
| `right_wrist_view` | TBD | |

## Controller TCP

| Parameter | Value |
|-----------|-------|
| IP | 192.168.77.58 |
| Port | 57770 |
| Connection mode | Client (driver connects to controller) |

## 3D Perception API

| Parameter | Value |
|-----------|-------|
| Method | TBD (function call / ROS topic / HTTP) |
| Output format | `{"object_name": [x, y, z]}` or TBD |
