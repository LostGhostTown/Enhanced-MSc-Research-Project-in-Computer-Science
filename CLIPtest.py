import sys
sys.path.append("D:\\consisid\\longCLIP\\Long-CLIP-main")  
sys.path.append("C:\\Users\\54637\\miniconda3\\envs\\consisid_new\\Lib\\site-packages")
import torch
import cv2
import numpy as np
from PIL import Image
# 官方标准导入方式
from model import longclip

def video_to_frames(video_path, num_frames=10):
    # （保持原拆帧逻辑不变）
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件：{video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_indices = np.linspace(0, total_frames-1, num_frames, dtype=int) if total_frames >= num_frames else list(range(total_frames))
    frames = []
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        else:
            print(f"警告：第 {idx} 帧读取失败")
    cap.release()
    return frames

def compute_video_longclip_score(video_path, prompt, model_path="../longCLIP/Long-CLIP-main/checkpoints/longclip-L.pt", num_frames=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备：{device}")
    
    # 官方标准加载方式（指定权重文件路径）
    print(f"加载Long-CLIP模型：{model_path}...")
    model, preprocess = longclip.load(model_path, device=device)
    print("✅ 模型加载完成（原生支持248 Token长文本）")
    
    frames = video_to_frames(video_path, num_frames)
    if len(frames) == 0:
        raise ValueError("未提取到有效视频帧")
    
    # 图像预处理与批量编码
    batch_frames = torch.cat([preprocess(frame).unsqueeze(0) for frame in frames], dim=0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(batch_frames)
    
    # 长文本编码（官方tokenize接口）
    text = longclip.tokenize([prompt]).to(device)
    print(f"提示词Token数：{text.shape[1]-2}（模型支持最大248）")
    with torch.no_grad():
        text_features = model.encode_text(text)
    
    # 计算CLIPScore
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    cos_sim = torch.matmul(image_features, text_features.T).squeeze(1)
    frame_scores = torch.clamp(100 * cos_sim, min=0).cpu().numpy()
    
    return {
        "frame_scores": frame_scores.tolist(),
        "avg_score": round(np.mean(frame_scores), 2),
        "median_score": round(np.median(frame_scores), 2),
        "max_score": round(np.max(frame_scores), 2)
    }

# 测试运行
if __name__ == "__main__":
    video_path = "D:\\consisid\\ConsisID\\finalresult\\oldVideo\\0007.mp4"
    prompt = "A woman stands in a sunlit flower shop, arranging a bouquet of peonies. Her fingers brush off a stray petal from her apron, and she leans in to smell a pink bloom—her eyes softening as the scent lingers. A small radio on the counter plays a folk song, and water droplets from the flower vases glisten on the wooden tabletop."
    try:
        result = compute_video_longclip_score(
            video_path=video_path,
            prompt=prompt,
            model_path="D:\\consisid\\longCLIP\\Long-CLIP-main\\checkpoints\\longclip-L.pt",  # 对应下载的权重文件
            num_frames=16
        )
        print("\n" + "="*70)
        print(f"Video path:{video_path}")
        print(f"Sampled frame number: {len(result['frame_scores'])}")
        print(f"Video average score: {result['avg_score']}")
        print("="*70)
    except Exception as e:
        print(f"❌ 运行失败：{str(e)}")