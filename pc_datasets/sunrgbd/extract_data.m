% Copyright (c) Facebook, Inc. and its affiliates.
%
% This source code is licensed under the MIT license found in the
% LICENSE file in the root directory of this source tree.

%% Dump train/val split.
% Author: Charles R. Qi
% Modified by Zijing Zhao (Peking University) at 2023

home_path = '/home/zhaozj/Datasets/sunrgbd/';

% Split train/val set.

addpath([home_path, 'SUNRGBDtoolbox']);
%% Construct Hash Map
hash_train = java.util.Hashtable;
hash_val = java.util.Hashtable;
split = load([home_path, 'SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat']);
N_train = length(split.alltrain);
N_val = length(split.alltest);
for i = 1:N_train
    folder_path = split.alltrain{i};
    folder_path(1:16) = '';
    hash_train.put(folder_path,0);
end
for i = 1:N_val
    folder_path = split.alltest{i};
    folder_path(1:16) = '';
    hash_val.put(folder_path,0);
end
%% Map data to train or val set.
load([home_path, 'SUNRGBDMeta3DBB_v2.mat']);
%% Create folder
folder = [home_path, 'sunrgbd_trainval'];
if ~exist(folder, 'dir')
    mkdir(folder);
end
fid_train = fopen([home_path, 'sunrgbd_trainval/train_data_idx.txt'], 'w');
fid_val = fopen([home_path, 'sunrgbd_trainval/val_data_idx.txt'], 'w');
disp(['Writing train/val split to: ', folder]);
for imageId = 1:10335
    data = SUNRGBDMeta(imageId);
    depthpath = data.depthpath;
    depthpath(1:16) = '';
    [filepath,name,ext] = fileparts(depthpath);
    [filepath,name,ext] = fileparts(filepath);
    if hash_train.containsKey(filepath)
        fprintf(fid_train, '%d\n', imageId);
    elseif hash_val.containsKey(filepath)
        fprintf(fid_val, '%d\n', imageId);
    else
        a = 1;
    end
end
fclose(fid_train);
fclose(fid_val);
disp('Splitting train/val set finished.');

% Extract data and generate v2 annotation.

addpath([home_path, 'SUNRGBDtoolbox/readData'])
load([home_path, 'SUNRGBDMeta2DBB_v2.mat']);
%% Create folders
depth_folder = [home_path, 'sunrgbd_trainval/depth/'];
if ~exist(depth_folder, 'dir')
    mkdir(depth_folder);
end
image_folder = [home_path, 'sunrgbd_trainval/image/'];
if ~exist(image_folder, 'dir')
    mkdir(image_folder);
end
calib_folder = [home_path, 'sunrgbd_trainval/calib/'];
if ~exist(calib_folder, 'dir')
    mkdir(calib_folder);
end
det_label_folder = [home_path, 'sunrgbd_trainval/label_v2/'];
if ~exist(det_label_folder, 'dir')
    mkdir(det_label_folder);
end
%% Read data
parfor imageId = 1:10335
    fprintf('Extracting data and v2 annotation: imageId: %d / 10355\n', 10355 - imageId);
    try
        data = SUNRGBDMeta(imageId);
        data.depthpath(1:17) = '';
        data.depthpath = strcat(home_path, data.depthpath);
        data.rgbpath(1:17) = '';
        data.rgbpath = strcat(home_path, data.rgbpath);
        % Write point cloud in depth map
        [rgb,points3d,depthInpaint,imsize]=read3dPoints(data);
        rgb(isnan(points3d(:,1)),:) = [];
        points3d(isnan(points3d(:,1)),:) = [];
        points3d_rgb = [points3d, rgb];
        % MAT files are 3x smaller than TXT files. In Python we can use
        % scipy.io.loadmat('xxx.mat')['points3d_rgb'] to load the data.
        mat_filename = strcat(num2str(imageId,'%06d'), '.mat');
        txt_filename = strcat(num2str(imageId,'%06d'), '.txt');
        parsave(strcat(depth_folder, mat_filename), points3d_rgb);
        % Write images
        copyfile(data.rgbpath, sprintf('%s/%06d.jpg', image_folder, imageId));
        % Write calibration
        dlmwrite(strcat(calib_folder, txt_filename), data.Rtilt(:)', 'delimiter', ' ');
        dlmwrite(strcat(calib_folder, txt_filename), data.K(:)', 'delimiter', ' ', '-append');
        % Write 2D and 3D box label
        data2d = SUNRGBDMeta2DBB(imageId);
        fid = fopen(strcat(det_label_folder, txt_filename), 'w');
        for j = 1:length(data.groundtruth3DBB)
            centroid = data.groundtruth3DBB(j).centroid;
            classname = data.groundtruth3DBB(j).classname;
            orientation = data.groundtruth3DBB(j).orientation;
            coeffs = abs(data.groundtruth3DBB(j).coeffs);
            box2d = data2d.groundtruth2DBB(j).gtBb2D;
            assert(strcmp(data2d.groundtruth2DBB(j).classname, classname));
            fprintf(fid, '%s %d %d %d %d %f %f %f %f %f %f %f %f\n', classname, box2d(1), box2d(2), box2d(3), box2d(4), centroid(1), centroid(2), centroid(3), coeffs(1), coeffs(2), coeffs(3), orientation(1), orientation(2));
        end
        fclose(fid);
    catch
    end
end

% Generate v1 annotation

%% V1 2D&3D BB
load([home_path, 'SUNRGBDtoolbox/Metadata/SUNRGBDMeta.mat']);
%% Create folders
det_label_folder = [home_path, 'sunrgbd_trainval/label_v1/'];
if ~exist(det_label_folder, 'dir')
    mkdir(det_label_folder);
end
%% Read
for imageId = 1:10335
    % imageId
    fprintf('Extracting v1 annotation: imageId: %d / 10355\n', imageId);
    try
        data = SUNRGBDMeta(imageId);
        data.depthpath(1:16) = '';
        data.depthpath = strcat([home_path, 'SUNRGBD'], data.depthpath);
        data.rgbpath(1:16) = '';
        data.rgbpath = strcat([home_path, 'SUNRGBD'], data.rgbpath);
        % MAT files are 3x smaller than TXT files. In Python we can use
        % scipy.io.loadmat('xxx.mat')['points3d_rgb'] to load the data.
        mat_filename = strcat(num2str(imageId,'%06d'), '.mat');
        txt_filename = strcat(num2str(imageId,'%06d'), '.txt');
        % Write 2D and 3D box label
        data2d = data;
        fid = fopen(strcat(det_label_folder, txt_filename), 'w');
        for j = 1:length(data.groundtruth3DBB)
            centroid = data.groundtruth3DBB(j).centroid;
            classname = data.groundtruth3DBB(j).classname;
            orientation = data.groundtruth3DBB(j).orientation;
            coeffs = abs(data.groundtruth3DBB(j).coeffs);
            box2d = data2d.groundtruth2DBB(j).gtBb2D;
            fprintf(fid, '%s %d %d %d %d %f %f %f %f %f %f %f %f\n', classname, box2d(1), box2d(2), box2d(3), box2d(4), centroid(1), centroid(2), centroid(3), coeffs(1), coeffs(2), coeffs(3), orientation(1), orientation(2));
        end
        fclose(fid);
        catch
    end
end
exit()

function parsave(filename, instance)
    save(filename, 'instance');
end
