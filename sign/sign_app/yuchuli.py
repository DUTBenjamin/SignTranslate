# yuchuli.py
import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import mediapipe as mp  # 添加 Mediapipe 库用于关键点检测

# 初始化 Mediapipe 的手部和姿态检测器
mp_hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2)


# mp_pose = mp.solutions.pose.Pose(static_image_mode=True)

def load_frames_and_keypoints_from_directory(directory, frame_size=(256, 256, 3), data_augmentation=False):
    """
    从指定目录加载视频帧，并提取手部关键点和身体姿态特征。
    参数：
        directory (str): 包含视频帧的目录路径。
        frame_size (tuple): 输出帧的尺寸 (宽度, 高度, 通道数)。
        data_augmentation (bool): 是否应用数据增强。
    返回：
        frames (numpy.ndarray): 加载并预处理后的帧数组，形状为 (帧数, 高度, 宽度, 通道数)。
        keypoints (numpy.ndarray): 提取的手部和身体关键点特征，形状为 (帧数, 关键点特征维度)。
    """
    frames = []
    keypoints_list = []
    valid_extensions = ('.png', '.jpg', '.jpeg')

    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory {directory} does not exist")

    # 定义数据增强操作
    if data_augmentation:
        augmentations = transforms.Compose([
            transforms.RandomResizedCrop(size=(frame_size[0], frame_size[1]), scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=(0.7, 1.3), contrast=(0.7, 1.3), saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        augmentations = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    for filename in sorted(os.listdir(directory)):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(valid_extensions):
            # 使用 OpenCV 加载图像
            frame = cv2.imread(file_path)
            if frame is None:
                print(f"Warning: Unable to load image {file_path}. Skipping...")
                continue
            # 转换为 RGB 格式
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 转换为 PIL 图像格式
            pil_frame = Image.fromarray(frame)
            # 应用数据增强
            augmented_frame = augmentations(pil_frame)
            frames.append(augmented_frame)

            # 提取手部关键点和身体姿态
            image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # 关键点检测
            hands_results = mp_hands.process(image)
            # pose_results = mp_pose.process(image)
            hands_keypoints = []
            # 初始化固定长度的关键点数组（双手126维身体99维 ）
            total_keypoints = 126
            keypoints = [0.0] * total_keypoints  # 初始化为全0数组

            # ------------------------------------------------------------
            # 处理手部关键点（最多两只手）
            if hands_results.multi_hand_landmarks:
                hands = hands_results.multi_hand_landmarks
                # 只处理前两只手（防止超过预分配空间）
                for hand_idx, hand in enumerate(hands[:2]):
                    start = hand_idx * 63  # 每只手21关键点 * 3(x,y,z)
                    # 遍历21个关键点
                    for lm_idx, landmark in enumerate(hand.landmark):
                        if lm_idx >= 21:  # 防止索引越界
                            break
                        pos = start + lm_idx * 3
                        # 确保索引在0-125范围内
                        if pos + 2 < 126:  # 手部占前126维
                            keypoints[pos] = landmark.x
                            keypoints[pos + 1] = landmark.y
                            keypoints[pos + 2] = landmark.z

            # ------------------------------------------------------------
            # 处理身体姿态关键点（必须检查 pose_results.pose_landmarks 是否存在）
            # if pose_results.pose_landmarks:  # 修复变量名并检查有效性
            #    pose = pose_results.pose_landmarks.landmark  # 正确访问姿态关键点
            #    start_body = 126  # 身体关键点从第126位开始

            # 遍历33个身体关键点
            #    for lm_idx, landmark in enumerate(pose):
            #        if lm_idx >= 33:  # 防止索引越界
            #            break
            #        pos = start_body + lm_idx * 3
            # 确保索引在126-224范围内
            #        if pos + 2 < 225:  # 总维度225
            #           keypoints[pos] = landmark.x
            #          keypoints[pos+1] = landmark.y
            #         keypoints[pos+2] = landmark.z

            # 将处理后的关键点加入列表
            keypoints_list.append(torch.tensor(keypoints, dtype=torch.float32))

    if not frames:
        return None, None

    # 将列表转换为张量
    frames_tensor = torch.stack(frames)
    frames = frames_tensor.permute(0, 2, 3, 1).numpy()
    keypoints_tensor = torch.stack(keypoints_list).numpy()

    return frames, keypoints_tensor