# Converting SUN RGB-D dataset

**SUN RGB-D**

SUN RGB-D dataset is a large-scale dataset collected from real-world single RGB-D images, 
thus exhibiting low quality point clouds with obvious point omissions.

<table>
  <tr>
    <td><img src="https://raw.githubusercontent.com/JeremyZhao1998/JeremyZhao1998.github.io/master/images/2025-DAVoteNet/sunrgbd/01.gif" width="100%"></td>
    <td><img src="https://raw.githubusercontent.com/JeremyZhao1998/JeremyZhao1998.github.io/master/images/2025-DAVoteNet/sunrgbd/02.gif" width="100%"></td>
    <td><img src="https://raw.githubusercontent.com/JeremyZhao1998/JeremyZhao1998.github.io/master/images/2025-DAVoteNet/sunrgbd/03.gif" width="100%"></td>
    <td><img src="https://raw.githubusercontent.com/JeremyZhao1998/JeremyZhao1998.github.io/master/images/2025-DAVoteNet/sunrgbd/04.gif" width="100%"></td>
  </tr>
</table>

## Raw data preparation

Follow the instructions in the [SUN RGB-D website](https://rgbd.cs.princeton.edu/) to download the raw data files
(SUNRGBD.zip, SUNRGBDMeta2DBB_v2.mat, SUNRGBDMeta3DBB_v2.mat, SUNRGBDtoolbox.zip), and then unzip them.

Place the raw data files and annotations in the following folders:

```
<dataset_root>
    └─ sunrgbd
        └─ SUNRGBD
        └─ SUNRGBDMeta2DBB_v2.mat
        └─ SUNRGBDMeta3DBB_v2.mat
        └─ SUNRGBDtoolbox
```

## Extract RGBD images

**This process requires installing ``matlab``.**

Run the script ``extract_data.m`` by matlab, which will generate ``sunrgbd_trainval`` folder:

```
<dataset_root>
    └─ sunrgbd
        └─ SUNRGBD
        └─ SUNRGBDMeta2DBB_v2.mat
        └─ SUNRGBDMeta3DBB_v2.mat
        └─ SUNRGBDtoolbox
        └─ sunrgbd_trainval
            └─ calib
            └─ depth
            └─ image
            └─ label_v1
            └─ label_v2
            └─ train_data_idx.txt
            └─ val_data_idx.txt
```

## Convert to point clouds

Run the following command to generate point cloud with axis aligned bounding boxes:

```bash
python convert_sunrgbd.py --dataset_root <dataset_root>/sunrgbd --axis_aligned 0
```

Run the following command to generate point cloud with rotated bounding boxes:

```bash
python convert_sunrgbd.py --dataset_root <dataset_root>/sunrgbd --axis_aligned 1
```

The generated data will be stored in ``<dataset_root>/sunrgbd/``:

```
<dataset_root>
    └─ sunrgbd
        └─ SUNRGBD
        └─ SUNRGBDMeta2DBB_v2.mat
        └─ SUNRGBDMeta3DBB_v2.mat
        └─ SUNRGBDtoolbox
        └─ sunrgbd_trainval
            └─ calib
            └─ depth
            └─ image
            └─ label_v1
            └─ label_v2
            └─ train_data_idx.txt
            └─ val_data_idx.txt
        └─ pc_bboxes_axis_aligned_train
        └─ pc_bboxes_axis_aligned_val
        └─ pc_bboxes_train
        └─ pc_bboxes_val
```
