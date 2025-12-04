The file structure should look like this:
```
root
├── data
│   └── KIT
│       ├── 3/
│       ├── ...
│       └── 1747/
│           ├── displace_from_left_to_right_01_poses.npz
│           ├── displace_from_left_to_right_02_poses.npz
│           └── ...
│   └── smpl
│       └── SMPL_NEUTRAL.pkl
│   └── initial_pose
│       ├── initial_pose_train.pkl
│       └── initial_pose_test.pkl
│   └── dataset
│       ├── kit_train_motion_dict.pkl
│       ├── kit_test_motion_dict.pkl
│       ├── kit_train_keys.txt
│       └── kit_test_keys.txt
```