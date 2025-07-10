import torch
import torch.nn as nn
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
from .CNNModel import CNNModel  # 导入自定义的CNN模型


class GestureRecognizer:
    def __init__(self, model_path, classes_dict):
        # 加载模型
        self.model = CNNModel()  # 使用自定义的CNNModel
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        # 存储类别映射
        self.classes = classes_dict
        self.reverse_classes = {v: k for k, v in classes_dict.items()}

        # 初始化MediaPipe手势检测
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.2
        )

    def preprocess_landmarks(self, landmarks, image_shape):
        """处理关键点数据，与realTime.py中的逻辑一致"""
        if not landmarks:
            return None, None

        # 提取坐标
        x_coords = []
        y_coords = []
        z_coords = []
        data = {}

        for i, landmark in enumerate(landmarks.landmark):
            x_coords.append(landmark.x)
            y_coords.append(landmark.y)
            z_coords.append(landmark.z)

        # 应用Min-Max归一化
        for i, landmark in enumerate(mp.solutions.hands.HandLandmark):
            lm = landmarks.landmark[i]
            data[f'{landmark.name}_x'] = lm.x - min(x_coords)
            data[f'{landmark.name}_y'] = lm.y - min(y_coords)
            data[f'{landmark.name}_z'] = lm.z - min(z_coords)

        # 创建边界框
        height, width = image_shape[:2]
        x_min = int(min(x_coords) * width) - 10
        y_min = int(min(y_coords) * height) - 10
        x_max = int(max(x_coords) * width) - 10
        y_max = int(max(y_coords) * height) - 10
        bbox = (x_min, y_min, x_max, y_max)

        return data, bbox

    def recognize_gesture(self, image):
        """识别手势，返回手势类别、置信度和边界框"""
        # 转换图像为RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        if not results.multi_hand_landmarks:
            return None, None, None

        # 只处理检测到的第一只手
        hand_landmarks = results.multi_hand_landmarks[0]

        # 预处理关键点
        landmarks_data, bbox = self.preprocess_landmarks(hand_landmarks, image.shape)
        if landmarks_data is None:
            return None, None, None


        # 根据mediapipe.HandLandmark的顺序，依次取出每个关键点的x,y,z
        feature_vector = []
        for landmark in mp.solutions.hands.HandLandmark:
            feature_vector.append(landmarks_data[f'{landmark.name}_x'])
            feature_vector.append(landmarks_data[f'{landmark.name}_y'])
            feature_vector.append(landmarks_data[f'{landmark.name}_z'])

        # 转换为numpy数组并调整形状为(1, 63, 1)
        feature_vector = np.array(feature_vector).reshape(1, 63, 1).astype(np.float32)
        input_tensor = torch.from_numpy(feature_vector).float()

        # 模型预测
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        # 获取手势标签
        gesture = self.reverse_classes[predicted_idx.item()]

        return gesture, confidence.item(), bbox