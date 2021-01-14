'''this is docstring'''
#!/usr/bin/python
# -*- coding: UTF-8 -*-

# main file for this project
# Copyright (C) 2021, David Shu. All rights reserved.
#
# Use of this source code is governed by a GPL license
# Author: David Shu(a294562476@gmail.com)

import argparse
import os
import re
import numpy as np
import pandas as pd
import torch
import cv2

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses
from val import normalize, pad_width


class VideoReader(object):
    """ this is docstring """
    def __init__(self, file):
        self.file_name = file
        self.cap = cv2.VideoCapture(self.file_name)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        # trans_img = cv2.transpose(img)
        # img = cv2.flip(trans_img, 1)
        if not was_read:
            raise StopIteration
        return img


def infer_fast(model, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=(128, 128, 128), img_scale=1/256):
    """[summary]

    Args:
        model ([type]): [description]
        img ([type]): [description]
        net_input_height_size ([type]): [description]
        stride ([type]): [description]
        upsample_ratio ([type]): [description]
        cpu ([type]): [description]
        pad_value (tuple, optional): [description]. Defaults to (0, 0, 0).
        img_mean (tuple, optional): [description]. Defaults to (128, 128, 128).
        img_scale ([type], optional): [description]. Defaults to 1/256.

    Returns:
        [type]: [description]
    """
    height, _, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cuda()

    stages_output = model(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio,
    interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio,
    interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad


def run_demo(model, image_provider, height_size, cpu, track, smooth, file):
    """[summary]

    Args:
        model ([type]): [description]
        image_provider ([type]): [description]
        height_size ([type]): [description]
        cpu ([type]): [description]
        track ([type]): [description]
        smooth ([type]): [description]
        file ([type]): [description]

    Returns:
        [type]: [description]
    """
    model = model.eval()
    if not cpu:
        model = model.cuda()

    point_list = []
    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    previous_poses = []

    # 保存视频
    fps = image_provider.fps
    width = image_provider.width
    height = image_provider.height
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    # video_saver = cv2.VideoWriter('TESTV.mp4', fourcc, fps, (height, width))
    save_video_path = os.path.join(os.getcwd(), 'video_output')
    if not os.path.exists(save_video_path):
        os.mkdir(save_video_path)
    save_video_name = os.path.join(save_video_path, file + '.mp4')
    video_saver = cv2.VideoWriter(save_video_name, fourcc, fps, (width, height))

    for img in image_provider:
        orig_img = img.copy()
        heatmaps, pafs, scale, pad = infer_fast(model, img, height_size, stride,
        upsample_ratio, cpu)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx],
            all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs, demo=True)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride
            / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride
            / upsample_ratio - pad[0]) / scale
        current_poses = []
        for pose_entry in pose_entries:
            if len(pose_entry) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entry[kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entry[kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entry[kpt_id]), 1])
            pose = Pose(pose_keypoints, pose_entry[18])
            current_poses.append(pose)

            # save keypoint in list
            key_point_list = pose_keypoints.flatten().tolist()
            point_list.append(key_point_list)

        if track:
            track_poses(previous_poses, current_poses, smooth=smooth)
            previous_poses = current_poses
        for pose in current_poses:
            pose.draw(img)
        img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
        for pose in current_poses:
            cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                          (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
            if track:
                cv2.putText(img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
        video_saver.write(img)
    return point_list


# 保存关键点
def save_point(point_list, file):
    """[summary]

    Args:
        point_list ([type]): [description]
        file ([type]): [description]
    """
    kpt_names = ['nose', 'neck', 'r_shoulder', 'r_elbow', 'r_wrist', 'l_shoulder',
                    'l_elbow', 'l_wrist', 'r_hip', 'r_knee', 'r_ankle', 'l_hip',
                    'l_knee', 'l_ankle', 'r_eye', 'l_eye', 'r_ear', 'l_ear']

    kpt_names_xy = []
    for i in kpt_names:
        kpt_names_x = i+'_x'
        kpt_names_xy.append(kpt_names_x)
        kpt_names_y = i+'_y'
        kpt_names_xy.append(kpt_names_y)

    data_frame = pd.DataFrame(point_list, columns=kpt_names_xy)
    save_excel_path = os.path.join(os.getcwd(), 'excel_output')
    if not os.path.exists(save_excel_path):
        os.mkdir(save_excel_path)
    save_excel_name = os.path.join(save_excel_path, file + '.xlsx')
    data_frame.to_excel(save_excel_name, index=False)


def file_name_mask(file, i):
    """[summary]

    Args:
        file ([type]): [description]
        i ([type]): [description]

    Returns:
        [type]: [description]
    """
    file_name_split_list = re.split(r'，|_|\.', file)
    if file_name_split_list[3] == 'right':
        file_name_split_list[3] = '1'
    else:
        file_name_split_list[3] = '0'
    file_name_result = (str(i).zfill(6) + '_' + file_name_split_list[1] + '_'
        + file_name_split_list[3])
    return file_name_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Lightweight human pose estimation python demo.
                       This is just for quick results preview.
                       Please, consider c++ demo for the best performance.''')
    parser.add_argument('--checkpoint-path', type=str, help='path to the checkpoint',
    default='checkpoints/checkpoint_iter_370000.pth')
    parser.add_argument('--height-size', type=int, default=256,
    help='network input layer height size')
    parser.add_argument('--video', type=str, default='data', help='path to video file or camera id')
    parser.add_argument('--cpu', action='store_true', help='run network inference on cpu')
    parser.add_argument('--track', type=int, default=1, help='track pose id in video')
    parser.add_argument('--smooth', type=int, default=1, help='smooth pose keypoints')
    args = parser.parse_args()

    if args.video == '' and args.images == '':
        raise ValueError('Either --video or --image has to be provided')

    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    load_state(net, checkpoint)
    file_name_list = os.listdir(args.video)
    for index, file_name in enumerate(file_name_list):
        file_name_dir = os.path.join(args.video, file_name)
        frame_provider = VideoReader(file_name_dir)
        file_name_mask_str = file_name_mask(file_name, index)
        print(file_name_mask_str)
        point_list_result = run_demo(net, frame_provider, args.height_size, args.cpu,
        args.track, args.smooth, file_name_mask_str)
        save_point(point_list_result, file_name_mask_str)
