## Indoor 3D object detection datasets

We provide unified format of existing indoor 3D object detection datasets **ScanNet**, **SUN RGB-D**, **3D Front**
as well as our newly proposed large-scale datasets **ProcTHOR-OD** and **ProcFront**.
Different datasets exhibit different distribution of style, point cloud quality, layout and instance features.

### Our proposed datasets

We open source the generation code of our proposed ProcTHOR-OD and ProcFront dataset at:
[ProcTHOR-OD](https://github.com/JeremyZhao1998/ProcTHOR-OD)
which provides the code of generating the ProcTHOR-OD layouts and the method to export 3D mesh files.
With our provided code, users can generate unlimited number of room layouts and export 3D mesh files.

**ProcTHOR-OD**

Our proposed ProcTHOR-OD dataset is a large-scale synthetic dataset for object detection in 3D.
It uses ProcTHOR generation pipeline to automatically generate 3D single room layouts, 
with accurate annotations of objects and their poses for object detection task.

<table>
  <tr>
    <td><img src="../figures/procthor/01.gif" width="100%"></td>
    <td><img src="../figures/procthor/02.gif" width="100%"></td>
    <td><img src="../figures/procthor/03.gif" width="100%"></td>
    <td><img src="../figures/procthor/04.gif" width="100%"></td>
  </tr>
</table>

**ProcFront**

ProcFront shares the same room layouts with ProcTHOR, but integrates instances from 3D Front dataset
to isolate the domain gap of layout and instance for domain adaptation investigations.

<table>
  <tr>
    <td><img src="../figures/procfront/01.gif" width="100%"></td>
    <td><img src="../figures/procfront/02.gif" width="100%"></td>
    <td><img src="../figures/procfront/03.gif" width="100%"></td>
    <td><img src="../figures/procfront/04.gif" width="100%"></td>
  </tr>
</table>

### Existing datasets

**ScanNet**

ScanNet dataset is a high quality real-world dataset collected from 3D scanners.

<table>
  <tr>
    <td><img src="../figures/scannet/01.gif" width="100%"></td>
    <td><img src="../figures/scannet/02.gif" width="100%"></td>
    <td><img src="../figures/scannet/03.gif" width="100%"></td>
    <td><img src="../figures/scannet/04.gif" width="100%"></td>
  </tr>
</table>

**SUN RGB-D**

SUN RGB-D dataset is a large-scale dataset collected from real-world single RGB-D images, 
thus exhibiting low quality point clouds with obvious point omissions.

<table>
  <tr>
    <td><img src="../figures/sunrgbd/01.gif" width="100%"></td>
    <td><img src="../figures/sunrgbd/02.gif" width="100%"></td>
    <td><img src="../figures/sunrgbd/03.gif" width="100%"></td>
    <td><img src="../figures/sunrgbd/04.gif" width="100%"></td>
  </tr>
</table>

**3D Front**

3D Front dataset is a synthetic dataset constructed by placing synthetic 3D models into rooms by
human expert designers.
It contains uniformly sampled high-quality 3D point clouds, but still lacks extensibility and realism.

<table>
  <tr>
    <td><img src="../figures/3dfront/01.gif" width="100%"></td>
    <td><img src="../figures/3dfront/02.gif" width="100%"></td>
    <td><img src="../figures/3dfront/03.gif" width="100%"></td>
    <td><img src="../figures/3dfront/04.gif" width="100%"></td>
  </tr>
</table>
