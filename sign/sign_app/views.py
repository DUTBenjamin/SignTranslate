from django.shortcuts import render
from django.http import HttpResponse, JsonResponse,FileResponse
from django.views.decorators.csrf import csrf_exempt
import base64
import cv2
import numpy as np
import os
import json
from django.shortcuts import render
from .gesture_model import GestureRecognizer
from sign_Project.settings import BASE_DIR
import jieba
import re
from datetime import datetime
from django.conf import settings
import tempfile
import cv2
import numpy as np
from PIL import Image, ImageSequence
import os
import tempfile
import difflib
import subprocess
from .models import SignHistory
from django.views.decorators.http import require_GET
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import mediapipe as mp
from .CNN_LSTM import FeatureExtractor, MultiModalCNNTransformerModel
from typing import Optional, List

# 连续手语识别模型全局变量
continuous_model = None
continuous_feature_extractor = None
continuous_label_map = None
continuous_mp_hands = None


def load_continuous_model():
    """加载连续手语识别模型"""
    global continuous_model, continuous_feature_extractor, continuous_label_map, continuous_mp_hands

    if continuous_model is None:
        try:
            model_path = os.path.join(BASE_DIR, 'ctcn_2', 'models', 'best_model.pth')
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

            # 初始化特征提取器
            continuous_feature_extractor = FeatureExtractor(model_name='resnet18')
            continuous_feature_extractor.eval()

            # 初始化Transformer模型
            continuous_model = MultiModalCNNTransformerModel(
                feature_dim=638,
                num_classes=len(checkpoint['index_to_label']),
                heads=2
            )
            continuous_model.load_state_dict(checkpoint['model_state_dict'])
            continuous_model.eval()

            # 加载标签映射
            continuous_label_map = {int(k): v for k, v in checkpoint['index_to_label'].items()}

            # 初始化Mediapipe手部检测
            continuous_mp_hands = mp.solutions.hands.Hands(
                static_image_mode=True,
                max_num_hands=2
            )

            print("连续手语识别模型加载成功")
        except Exception as e:
            print(f"加载连续手语识别模型失败: {str(e)}")


# 在应用启动时加载模型
load_continuous_model()


def preprocess_frame(frame):
    """预处理单帧图像"""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    return transform(pil_image)


def extract_keypoints(frame, mp_hands):
    """提取关键点"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hands_results = mp_hands.process(frame_rgb)

    # 初始化关键点数组（126维）
    keypoints = np.zeros(126, dtype=np.float32)

    if hands_results.multi_hand_landmarks:
        hands = hands_results.multi_hand_landmarks[:2]
        for hand_idx, hand in enumerate(hands):
            start = hand_idx * 63
            for lm_idx, landmark in enumerate(hand.landmark[:21]):
                pos = start + lm_idx * 3
                if pos + 2 < 126:
                    keypoints[pos] = landmark.x
                    keypoints[pos + 1] = landmark.y
                    keypoints[pos + 2] = landmark.z

    return torch.tensor(keypoints, dtype=torch.float32)



def tokenize_text(text):
    """分词函数，保留标点符号作为单独token"""
    # 用正则表达式分割文本，保留标点符号
    tokens = re.findall(r"[\w']+|[.,!?;]", text.lower())
    return tokens

def find_similar_word(word, word_list, cutoff=0.8):
    """查找相似词"""
    matches = difflib.get_close_matches(word, word_list, n=1, cutoff=cutoff)
    return matches[0] if matches else None

# 修改output_dir路径
output_dir = os.path.join(settings.BASE_DIR, 'static', 'sign_app', 'output')
os.makedirs(output_dir, exist_ok=True)

# 修改get_sign_file函数中的base_dir
def get_sign_file(word, word_list, base_dir=None):
    """获取手语图片文件路径（中英文统一，优先单词.png，找不到则字母.png）"""
    if base_dir is None:
        base_dir = os.path.join(settings.BASE_DIR, 'static', 'sign_app', 'words')
    if word in [',', '.', '!', '?', ';', ' ']:
        return None
    # 优先查找单词.png
    png_file = os.path.join(base_dir, f'{word}.png')
    if isinstance(png_file, str) and os.path.exists(png_file):
        return [png_file]
    # 查找相似词.png
    similar_word = find_similar_word(word, word_list)
    if similar_word:
        png_file = os.path.join(base_dir, f'{similar_word}.png')
        if isinstance(png_file, str) and os.path.exists(png_file):
            return [png_file]
    # 如果没有找到单词图片，使用字母拼写（每个字母.png）
    letter_dir = os.path.join(settings.BASE_DIR, 'static', 'sign_app', 'alphabet')
    letter_files = []
    for letter in word:
        letter_file = os.path.join(letter_dir, f'{letter}.png')
        if isinstance(letter_file, str) and os.path.exists(letter_file):
            letter_files.append(letter_file)
    return letter_files if letter_files else None

def tokenize_chinese_text(text: str) -> list:
    """
    对中文句子进行分词，并保留标点符号作为独立的元素。
    """
    tokens = []
    for fragment in re.split(r'([，。！？；])', text):
        if not fragment:
            continue
        if fragment in '，。！？；':
            tokens.append(fragment)
        else:
            tokens.extend(jieba.lcut(fragment))
    return [token for token in tokens if token.strip()]

# 中文到英文的映射表
CN2EN = {
    '你': 'you',
    '我': 'me',
    '他': 'he',
    '她': 'she',
    '它': 'it',
    '我们': 'we',
    '你们': 'you_all',
    '大家': 'everyone',
    '自己': 'self',
    '好': 'good',
    '喜欢': 'like',
    '爱': 'love',
    '明白': 'understand',
    '懂': 'understand',
    '知道': 'know',
    '认识': 'know_person',
    '觉得': 'think',
    '认为': 'think',
    '学习': 'study',
    '写': 'write',
    '读': 'read',
    '听': 'listen',
    '说': 'speak',
    '叫': 'call',
    '喊': 'shout',
    '买': 'buy',
    '卖': 'sell',
    '给': 'give',
    '要': 'want',
    '需要': 'need',
    '愿意': 'willing',
    '可以': 'can',
    '能': 'can',
    '能够': 'able',
    '会': 'will',
    '想': 'want_to',
    '做': 'do',
    '吃': 'eat',
    '喝': 'drink',
    '吃饭': 'eat_meal',
    '饭': 'meal',
    '来': 'come',
    '去': 'go',
    '在': 'at',
    '于': 'at',
    '当': 'when',
    '有': 'have',
    '不是': 'not',
    '是': 'yes',
    '那里': 'there',
    '这里': 'here',
    '昨天': 'yesterday',
    '今天': 'today',
    '明天': 'tomorrow',
    '后天': 'day_after_tomorrow',
    '明白': 'understand',
    '看': 'see',
    '您': 'you_polite',
}

def get_sign_image_paths(word: str, words_dir: str, chars_dir: str) -> Optional[List]:
    if word in '，。！？；':
        return []
    mapped = CN2EN.get(word, word)
    word_image_path = os.path.join(words_dir, f"{mapped}.png")
    if os.path.exists(word_image_path):
        return [word_image_path]
    if len(word) > 1:
        char_paths = []
        all_chars_found = True
        for char in word:
            mapped_char = CN2EN.get(char, char)
            char_image_path = os.path.join(chars_dir, f"{mapped_char}.png")
            if os.path.exists(char_image_path):
                char_paths.append(char_image_path)
            else:
                all_chars_found = False
                break
        if all_chars_found:
            return char_paths
    if len(word) == 1:
        mapped_char = CN2EN.get(word, word)
        char_image_path = os.path.join(chars_dir, f"{mapped_char}.png")
        if os.path.exists(char_image_path):
            return [char_image_path]
    return None

def convert_mp4_to_h264(input_path, output_path):
    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-an', output_path
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def create_sign_language_video_from_images(
    words: list,
    words_dir: str,
    chars_dir: str,
    output_dir: str,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    image_duration: float = 1
) -> Optional[str]:
    print(f"[调试] 分词结果: {words}")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f"sign_language_{timestamp}.mp4"
    output_video_path = os.path.join(output_dir, output_filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    black_frame = np.zeros((height, width, 3), dtype=np.uint8)
    for word in words:
        image_paths = get_sign_image_paths(word, words_dir, chars_dir)
        print(f"[调试] 词: {word}，图片路径: {image_paths}")
        if image_paths is None:
            print(f"[调试] 未找到图片: {word}，words_dir={words_dir}，chars_dir={chars_dir}")
            for _ in range(int(fps * image_duration)):
                out.write(black_frame)
            continue
        if not image_paths:
            print(f"[调试] 标点或空图片: {word}")
            for _ in range(int(fps * 0.5)):
                out.write(black_frame)
            continue
        for image_path in image_paths:
            try:
                img = cv2.imread(image_path)
                if img is None:
                    print(f"[调试] OpenCV读取失败，尝试用PIL读取: {image_path}")
                    try:
                        pil_img = Image.open(image_path).convert('RGB')
                        img = np.array(pil_img)[..., ::-1]  # PIL转BGR
                    except Exception as e:
                        print(f"[调试] PIL读取也失败: {image_path}, 错误: {e}")
                        for _ in range(int(fps * image_duration)):
                            out.write(black_frame)
                        continue
                resized_img = cv2.resize(img, (width, height))
                print(f"[DEBUG] resized_img.shape: {resized_img.shape}, dtype: {resized_img.dtype}")
                for _ in range(int(fps * image_duration)):
                    out.write(resized_img)
            except Exception as e:
                print(f"[调试] 处理图片异常: {image_path}, 错误: {e}")
                for _ in range(int(fps * image_duration)):
                    out.write(black_frame)
        for _ in range(int(fps * 0.2)):
            out.write(black_frame)
    out.release()
    # 转码
    h264_path = output_video_path.replace('.mp4', '_h264.mp4')
    convert_mp4_to_h264(output_video_path, h264_path)
    return h264_path

# 修改generate_animation，支持中英文分流
@csrf_exempt
def generate_animation(request):
    """
    处理动画生成请求，支持中英文切换
    """
    if request.method == 'POST':
        try:
            text = request.POST.get('text', '').strip()
            lang = request.POST.get('lang', 'en').strip()  # 默认为英文
            if not text:
                return JsonResponse({'error': '请输入要翻译的文本'}, status=400)
            if lang == 'zh':
                # 中文逻辑
                words_dir = os.path.join(settings.BASE_DIR, 'static', 'sign_app', 'words-zhcn')
                chars_dir = os.path.join(settings.BASE_DIR, 'static', 'sign_app', 'words-zhcn')
                output_dir = os.path.join(settings.BASE_DIR, 'static', 'sign_app', 'output')
                os.makedirs(output_dir, exist_ok=True)
                tokenized_words = tokenize_chinese_text(text)
                # 用图片合成视频
                output_video = create_sign_language_video_from_images(
                    words=tokenized_words,
                    words_dir=words_dir,
                    chars_dir=chars_dir,
                    output_dir=output_dir
                )
            else:
                # 英文逻辑
                words_dir = os.path.join(settings.BASE_DIR, 'static', 'sign_app', 'words')
                word_list = [os.path.splitext(f)[0] for f in os.listdir(words_dir) if f.endswith('.mp4') or f.endswith('.png')]
                words = tokenize_text(text)
                output_video = create_sign_language_sequence_en(words, word_list)
            if output_video and os.path.exists(output_video):
                media_output_dir = os.path.join(settings.MEDIA_ROOT, 'sign_app', 'output')
                os.makedirs(media_output_dir, exist_ok=True)
                import shutil
                video_filename = os.path.basename(output_video)
                media_video_path = os.path.join(media_output_dir, video_filename)
                shutil.copy2(output_video, media_video_path)
                video_url = f"{settings.MEDIA_URL}sign_app/output/{video_filename}"
                if not video_url.startswith(('http://', 'https://')):
                    video_url = request.build_absolute_uri(video_url)
                # 保存历史记录
                SignHistory.objects.create(text=text, video_file=video_url, lang=lang)
                return JsonResponse({'video_url': video_url})
            else:
                return JsonResponse({'error': '无法生成手语动画'}, status=500)
        except Exception as e:
            print(f"生成动画错误: {str(e)}")
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': '无效的请求方法'}, status=400)

def create_sign_language_sequence_en(words, word_list):
    """英文：支持mp4和png混合合成，输出为mp4格式"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    # 1. 先生成一个临时的、非网页兼容的视频
    temp_output_filename = f"temp_sign_language_{timestamp}.mp4"
    temp_video_path = os.path.join(output_dir, temp_output_filename)

    width, height = 640, 480
    fps = 24
    # OpenCV 使用 'mp4v' 创建初始视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
    
    # 2. 这是最终的、经过转码的、网页兼容的视频路径
    final_output_filename = f"sign_language_{timestamp}_h264.mp4"
    final_video_path = os.path.join(output_dir, final_output_filename)

    print(f"[DEBUG] 临时视频路径: {temp_video_path}")
    print(f"[DEBUG] out.isOpened(): {out.isOpened()}")

    black_frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    try:
        # --- 你原有的帧写入循环代码保持不变 ---
        for word in words:
            # ... (你原有的读取 mp4 和 png 并写入帧的逻辑)
            print(f"[DEBUG] 处理单词: {word}")
            sign_file = get_sign_file_en(word, word_list)
            print(f"[DEBUG] sign_file: {sign_file}")
            if isinstance(sign_file, str) and sign_file.endswith('.mp4'):
                cap = cv2.VideoCapture(sign_file)
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    out.write(cv2.resize(frame, (width, height)))
                cap.release()
            elif isinstance(sign_file, list) and sign_file:
                for image_path in sign_file:
                    img = cv2.imread(image_path)
                    if img is None:
                        for _ in range(fps): out.write(black_frame)
                        continue
                    resized_img = cv2.resize(img, (width, height))
                    for _ in range(fps): out.write(resized_img)
            else:
                for _ in range(fps): out.write(black_frame)
        # --- 帧写入循环结束 ---
        
    except Exception as e:
        print(f"生成临时视频出错: {str(e)}")
        out.release() # 确保即使出错也释放写入器
        return None
    finally:
        # 确保写入器总是被释放
        out.release()
        print(f"[DEBUG] 临时视频写入完成。")

    # --- 3. 关键的转码步骤 ---
    try:
        if not os.path.exists(temp_video_path) or os.path.getsize(temp_video_path) == 0:
            print("[ERROR] 临时视频文件未生成或为空，无法转码。")
            return None

        print(f"[DEBUG] 开始转码: 从 {temp_video_path} 到 {final_video_path}")
        convert_mp4_to_h264(temp_video_path, final_video_path)
        print(f"[DEBUG] 转码成功。最终视频大小: {os.path.getsize(final_video_path)} bytes")
        
        # 4. 清理临时的视频文件
        os.remove(temp_video_path)
        
        # 5. 返回转码后、网页兼容的视频路径
        return final_video_path
        
    except Exception as e:
        print(f"FFmpeg 转码失败: {str(e)}")
        # 如果转码失败，最好返回 None 表示失败
        return None

def get_sign_file_zh(word, base_dir=None):
    """中文：始终查找png，找不到就返回None"""
    if base_dir is None:
        base_dir = os.path.join(settings.BASE_DIR, 'static', 'sign_app', 'words-zhcn')
    if word in '，。！？； ':
        return None
    png_file = os.path.join(base_dir, f'{word}.png')
    if os.path.exists(png_file):
        return [png_file]
    # 逐字查找
    letter_files = []
    for char in word:
        char_file = os.path.join(base_dir, f'{char}.png')
        if os.path.exists(char_file):
            letter_files.append(char_file)
    return letter_files if letter_files else None

def get_sign_file_en(word, word_list, base_dir=None):
    """英文：优先查找mp4，没有则查png，再没有查每个字母的png"""
    if base_dir is None:
        base_dir = os.path.join(settings.BASE_DIR, 'static', 'sign_app', 'words')
    if word in [',', '.', '!', '?', ';', ' ']:
        return None
    # 优先查找单词.mp4
    mp4_file = os.path.join(base_dir, f'{word}.mp4')
    if os.path.exists(mp4_file):
        return mp4_file
    # 查找相似词.mp4
    similar_word = find_similar_word(word, word_list)
    if similar_word:
        mp4_file = os.path.join(base_dir, f'{similar_word}.mp4')
        if os.path.exists(mp4_file):
            return mp4_file
    # 查找单词.png
    png_file = os.path.join(base_dir, f'{word}.png')
    if os.path.exists(png_file):
        return [png_file]
    # 查找相似词.png
    if similar_word:
        png_file = os.path.join(base_dir, f'{similar_word}.png')
        if os.path.exists(png_file):
            return [png_file]
    # 查找每个字母.png
    letter_dir = os.path.join(settings.BASE_DIR, 'static', 'sign_app', 'alphabet')
    letter_files = []
    for letter in word:
        letter_file = os.path.join(letter_dir, f'{letter}.png')
        if os.path.exists(letter_file):
            letter_files.append(letter_file)
    return letter_files if letter_files else None


'''下面为手语翻译的代码'''
# 手势类别字典（与训练时一致）
classes_dict = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8,
    'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16,
    'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25
}

# 加载手势的详细信息
gesture_info_path = os.path.join(BASE_DIR, 'model', 'gesture_labels.json')
try:
    # 指定使用 UTF-8 编码打开文件
    with open(gesture_info_path, 'r', encoding='utf-8') as f:
        gesture_info_by_index = json.load(f)

    # 创建按字母符号索引的字典
    gesture_info = {}
    for index, info in gesture_info_by_index.items():
        symbol = info['symbol']
        gesture_info[symbol] = {
            'name': info['name'],
            'description': info['description']
        }

    # 添加未检测到手势的情况
    gesture_info['None'] = {'name': '未检测到手势', 'description': '请确保手在摄像头范围内'}

except FileNotFoundError:
    # 如果文件不存在，创建默认的手势信息
    print("手势信息文件未找到，使用默认信息")
    gesture_info = {letter: {'name': f'字母 {letter}', 'description': f'这是字母 {letter} 的手势'}
                    for letter in classes_dict}
    gesture_info['None'] = {'name': '未检测到手势', 'description': '请确保手在摄像头范围内'}
except Exception as e:
    # 添加额外的错误处理
    print(f"加载手势信息文件时出错: {str(e)}")
    gesture_info = {letter: {'name': letter, 'description': f'手势 {letter}'}
                    for letter in classes_dict}
    gesture_info['None'] = {'name': '未检测到手势', 'description': '请确保手在摄像头范围内'}

# 初始化手势识别器
model_path = os.path.join(os.path.dirname(__file__), '../model/CNN_model_alphabet_SIBI.pth')
recognizer = GestureRecognizer(model_path, classes_dict)


def realtime_view(request):
    """渲染实时识别页面"""
    return render(request, 'sign_app/realtime.html')

@csrf_exempt
def get_gesture_info(request):
    """返回手势信息"""
    return JsonResponse(gesture_info)


@csrf_exempt
def recognize_gesture(request):
    """处理手势识别请求"""
    if request.method == 'POST':
        try:
            # 获取Base64编码的图像数据
            data = request.POST.get('image')
            if not data:
                return JsonResponse({'error': '未提供图像数据'}, status=400)

            # 移除Base64前缀
            if 'base64,' in data:
                data = data.split('base64,')[1]

            # 解码Base64图像
            img_data = base64.b64decode(data)
            nparr = np.frombuffer(img_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # 识别手势
            gesture_symbol, confidence, bbox = recognizer.recognize_gesture(image)

            # 获取手势的详细信息
            if gesture_symbol is None:
                info = gesture_info['None']
                return JsonResponse({
                    'gesture': 'None',
                    'name': info['name'],
                    'description': info['description'],
                    'confidence': 0.0
                })
            else:
                # 使用get方法并提供默认值
                info = gesture_info.get(gesture_symbol, {
                    'name': gesture_symbol,
                    'description': f'手势 {gesture_symbol}'
                })

                # 返回识别结果
                return JsonResponse({
                    'gesture': gesture_symbol,
                    'name': info['name'],
                    'description': info['description'],
                    'confidence': confidence,
                    'bbox': bbox  # (x_min, y_min, x_max, y_max)
                })

        except Exception as e:
            # 打印详细错误信息到控制台
            import traceback
            traceback.print_exc()
            return JsonResponse({'error': str(e)}, status=500)


    return JsonResponse({'error': '无效的请求方法'}, status=400)
print(f"Received request at /sign/recognize/")


def continuous_view(request):
    """连续手语翻译页面"""
    return render(request, 'sign_app/continuous.html')

def animation_view(request):
    """手语动画生成页面"""
    return render(request, 'sign_app/animation.html')

def technology_view(request):
    """技术原理页面"""
    return render(request, 'sign_app/technology.html')

def about_view(request):
    """关于我们页面"""
    return render(request, 'sign_app/about.html')

@require_GET
def get_animation_history(request):
    """返回最新10条动画历史记录"""
    records = SignHistory.objects.order_by('-created_at')[:10]
    data = [
        {
            'text': r.text,
            'video_file': r.video_file,
            'lang': r.lang,
            'created_at': r.created_at.strftime('%Y-%m-%d %H:%M:%S')
        } for r in records
    ]
    return JsonResponse({'history': data})

@csrf_exempt
def clear_history(request):
    if request.method == 'POST':
        from sign_app.models import SignHistory
        SignHistory.objects.all().delete()
        return JsonResponse({'success': True})
    return JsonResponse({'success': False, 'error': 'Invalid method'})

# 新增管理命令：清空历史记录和output视频
# 文件：sign_app/management/commands/clear_history_and_output.py


# 新增两个视图函数
def image_recognition_view(request):
    """中文手语图片识别页面"""
    return render(request, 'sign_app/image_recognition.html')

def video_recognition_view(request):
    """中文手语视频识别页面"""
    return render(request, 'sign_app/video_recognition.html')

# 图片识别处理
@csrf_exempt
def handle_video_recognition(request):
    if request.method == 'POST':
        # 处理图片识别逻辑
        if request.method == 'POST':
            # 确保模型已加载
            load_continuous_model()

            # 检查是否有文件上传
            if 'video' not in request.FILES:
                return JsonResponse({'error': '未上传视频文件'}, status=400)

            video_file = request.FILES['video']

            # 保存临时视频文件
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                for chunk in video_file.chunks():
                    tmp_file.write(chunk)
                tmp_file_path = tmp_file.name

            try:
                # 读取视频
                cap = cv2.VideoCapture(tmp_file_path)
                frames = []
                keypoints_list = []
                frame_count = 0
                sample_rate = 3  # 每3帧采样1帧

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # 采样帧
                    frame_count += 1
                    if frame_count % sample_rate != 0:
                        continue

                    # 预处理帧
                    processed_frame = preprocess_frame(frame)
                    frames.append(processed_frame)

                    # 提取关键点
                    keypoints = extract_keypoints(frame, continuous_mp_hands)
                    keypoints_list.append(keypoints)

                cap.release()

                if not frames:
                    return JsonResponse({'error': '未检测到有效视频帧'}, status=400)

                # 转换为Tensor并处理
                frames_tensor = torch.stack(frames)
                keypoints_tensor = torch.stack(keypoints_list)

                with torch.no_grad():
                    # 提取并拼接特征
                    visual_features = continuous_feature_extractor(
                        frames_tensor.view(-1, 3, 256, 256)
                    ).view(1, -1, 512)

                    combined_features = torch.cat([
                        visual_features,
                        keypoints_tensor.unsqueeze(0)
                    ], dim=2)

                    # 调整序列长度
                    seq_len = combined_features.shape[1]
                    if seq_len < 170:
                        padding = torch.zeros(1, 170 - seq_len, 638)
                        combined_features = torch.cat([combined_features, padding], dim=1)
                    else:
                        combined_features = combined_features[:, :170]

                    outputs = continuous_model(combined_features)
                    probs = torch.softmax(outputs, dim=1)

                # 处理预测结果
                pred_idx = torch.argmax(probs).item()
                label = continuous_label_map.get(pred_idx, "未知标签")
                confidence = probs[0][pred_idx].item()

                return JsonResponse({
                    'result': label,
                    'confidence': f"{confidence:.2%}",
                    'message': f"识别结果: {label} (置信度: {confidence:.2%})"
                })

            except Exception as e:
                return JsonResponse({'error': f"处理错误: {str(e)}"}, status=500)
            finally:
                # 删除临时文件
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)

        return JsonResponse({'error': '无效的请求方法'}, status=400)






# 视频识别处理
@csrf_exempt
def handle_image_recognition(request):
    if request.method == 'POST':
        # 处理视频识别逻辑
        load_continuous_model()

        # 检查是否有文件上传
        if 'images' not in request.FILES:
            return JsonResponse({'error': '未上传图片文件'}, status=400)

        # 获取上传的图片文件
        image_files = request.FILES.getlist('images')

        # 按文件名排序
        image_files.sort(key=lambda x: x.name)

        frames = []
        keypoints_list = []

        try:
            for image_file in image_files:
                # 将InMemoryUploadedFile转换为OpenCV图像
                image_data = image_file.read()
                nparr = np.frombuffer(image_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                # 预处理帧
                processed_frame = preprocess_frame(frame)
                frames.append(processed_frame)

                # 提取关键点
                keypoints = extract_keypoints(frame, continuous_mp_hands)
                keypoints_list.append(keypoints)

            if not frames:
                return JsonResponse({'error': '未检测到有效图片'}, status=400)

            # 转换为Tensor并处理
            frames_tensor = torch.stack(frames)
            keypoints_tensor = torch.stack(keypoints_list)

            with torch.no_grad():
                # 提取并拼接特征
                visual_features = continuous_feature_extractor(
                    frames_tensor.view(-1, 3, 256, 256)
                ).view(1, -1, 512)

                combined_features = torch.cat([
                    visual_features,
                    keypoints_tensor.unsqueeze(0)
                ], dim=2)

                # 调整序列长度
                seq_len = combined_features.shape[1]
                if seq_len < 170:
                    padding = torch.zeros(1, 170 - seq_len, 638)
                    combined_features = torch.cat([combined_features, padding], dim=1)
                else:
                    combined_features = combined_features[:, :170]

                outputs = continuous_model(combined_features)
                probs = torch.softmax(outputs, dim=1)

            # 处理预测结果
            pred_idx = torch.argmax(probs).item()
            label = continuous_label_map.get(pred_idx, "未知标签")
            confidence = probs[0][pred_idx].item()

            return JsonResponse({
                'result': label,
                'confidence': f"{confidence:.2%}",
                'message': f"识别结果: {label} (置信度: {confidence:.2%})"
            })

        except Exception as e:
            return JsonResponse({'error': f"处理错误: {str(e)}"}, status=500)

    return JsonResponse({'error': '无效的请求方法'}, status=400)
