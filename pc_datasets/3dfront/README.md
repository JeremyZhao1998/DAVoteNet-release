# Converting 3D Front dataset

**3D Front**

3D Front dataset is a synthetic dataset constructed by placing synthetic 3D models into rooms by
human expert designers.
It contains uniformly sampled high-quality 3D point clouds, but still lacks extensibility and realism.

The convertion code of 3D Front dataset is provided at: [3dfront](./3dfront/README.md)

<table>
  <tr>
    <td><img src="https://raw.githubusercontent.com/JeremyZhao1998/JeremyZhao1998.github.io/master/images/2025-DAVoteNet/3dfront/01.gif" width="100%"></td>
    <td><img src="https://raw.githubusercontent.com/JeremyZhao1998/JeremyZhao1998.github.io/master/images/2025-DAVoteNet/3dfront/02.gif" width="100%"></td>
    <td><img src="https://raw.githubusercontent.com/JeremyZhao1998/JeremyZhao1998.github.io/master/images/2025-DAVoteNet/3dfront/03.gif" width="100%"></td>
    <td><img src="https://raw.githubusercontent.com/JeremyZhao1998/JeremyZhao1998.github.io/master/images/2025-DAVoteNet/3dfront/04.gif" width="100%"></td>
  </tr>
</table>

## Raw data preparation

Follow the instructions in the [3D Front website](https://tianchi.aliyun.com/dataset/65347) to download the raw data files.
(Hint: you have to fill in the “3D-FRONT Terms of Use” and send email to apply for the dataset.)

Place the raw data files and annotations in the following folders:

```
<dataset_root>
    └─ 3dfront
        └─ 3D-FRONT
        └─ 3D-FRONT-texture
        └─ 3D-FUTURE-model
```

## Select scenes and convert to ply format

Run the following command to select scenes and convert to ply format:
```
python convert_3dfront_to_ply.py --dataset_root <dataset_root>/3dfront --out_dir <dataset_root>/3dfront
```

The ply files will be stored in ``<dataset_root>/3dfront/3dfront_pc_data``:

```
<dataset_root>
    └─ 3dfront
        └─ 3D-FRONT
        └─ 3D-FRONT-texture
        └─ 3D-FUTURE-model
        └─ 3dfront_pc_data
            └─ train
                └─ Bedroom-10003_retrieval.ply
                └─ ...
            └─ val
                └─ Bedroom-11202_retrieval.ply
                └─ ...
```

Note that the scene room and train/val split of 3D Front follows EchoScene and InstructScene:

```
Zhai G, Örnek E P, Chen D Z, et al. Echoscene: Indoor scene generation via information echo over scene graph diffusion[C]//European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2024: 167-184.
Lin C, Mu Y. Instructscene: Instruction-driven 3d indoor scene synthesis with semantic graph prior[J]. arXiv preprint arXiv:2402.04717, 2024.
```

## Convert to our unified format

Run the following command to convert to our unified format with axis aligned bounding boxes:
```
python convert_data.py --dataset_root <dataset_root>/3dfront --out_dir <dataset_root>/3dfront --axis_aligned 1
```

Run the following command to convert to our unified format with rotated bounding boxes:
```
python convert_data.py --dataset_root <dataset_root>/3dfront --out_dir <dataset_root>/3dfront --axis_aligned 0
```

The converted data will be stored in ``<dataset_root>/3dfront/``:

```
<dataset_root>
    └─ 3dfront
        └─ 3D-FRONT
        └─ 3D-FRONT-texture
        └─ 3D-FUTURE-model
        └─ 3dfront_pc_data
            └─ train
                └─ Bedroom-10003_retrieval.ply
                └─ ...
            └─ val
                └─ Bedroom-11202_retrieval.ply
                └─ ...
        └─ pc_bboxes_axis_aligned_train
        └─ pc_bboxes_axis_aligned_val
        └─ pc_bboxes_train
        └─ pc_bboxes_val
```
