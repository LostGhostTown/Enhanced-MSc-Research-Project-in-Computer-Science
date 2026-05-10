#========== Step 1: Minimally simulate Triton (completely avoid recursion) ==========
import sys
import os
import gc  
from unittest.mock import MagicMock

# 1. Directly create simple mock objects to avoid nested recursion
triton_mock = MagicMock()
# Add necessary attributes to the mock object to avoid PyTorch detection errors
triton_mock.__spec__ = MagicMock(name="triton")
triton_mock.language = MagicMock(__spec__=MagicMock(name="triton.language"))
triton_mock.compiler = MagicMock(__spec__=MagicMock(name="triton.compiler"))
triton_mock.runtime = MagicMock(__spec__=MagicMock(name="triton.runtime"))

# 2. Globally replace the triton module
sys.modules['triton'] = triton_mock
sys.modules['triton.language'] = triton_mock.language
sys.modules['triton.compiler'] = triton_mock.compiler
sys.modules['triton.runtime'] = triton_mock.runtime

# ========== Step 2: Add local model directory to sys.path ==========
MODEL_PATH = r"D:\cache\huggingface\modules\transformers_modules\THUDM\VisionReward-Video\VisionReward-Video"
sys.path.insert(0, MODEL_PATH)

# ========== Step 3: Skip transformers dependency checks ==========
try:
    from transformers import dynamic_module_utils
    def mock_check_imports(*args, **kwargs):
        return []
    dynamic_module_utils.check_imports = mock_check_imports
except (ImportError, AttributeError):
    pass

import io
import json
import numpy as np
import torch
from decord import cpu, VideoReader, bridge
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from tqdm import tqdm

# ===================== Path Configuration (matches your file structure) =====================
QUESTIONS_PATH = "VisionReward_Video/VisionReward_video_qa_select.txt"
WEIGHT_PATH = "VisionReward_Video/weight.json"

# ===================== Exact Dimension Mapping (matches your curated QA file) =====================
DIMENSION_INDEX_MAP = {
    "Alignment": [0, 1, 2],          
    "Stability": [20, 22],           
    "Preservation": [13, 14, 15, 16, 17],  
    "Physics": [21, 28]              
}

# Load QA questions and weights
with open(QUESTIONS_PATH, 'r', encoding='utf-8') as f:
    questions = [q.strip() for q in f.readlines()]

with open(WEIGHT_PATH, 'r', encoding='utf-8') as f:
    weight = np.array(json.load(f), dtype=np.float32)

# Device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# ===================== New: Core GPU Memory Cleanup Function =====================
def clear_gpu_memory():
    """
    Thoroughly clean up GPU VRAM:
    1. Delete model/Tokenizer objects
    2. Clear PyTorch CUDA cache
    3. Force system garbage collection
    Compatible with CPU/GPU environments, no errors
    """
    global model, tokenizer
    # 1. Release VRAM/memory occupied by model and tokenizer
    try:
        del model
        del tokenizer
    except NameError:
        pass
    # 2. Clear PyTorch CUDA cache (core)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()  # Clean up IPC shared memory
    # 3. Force Python garbage collection
    gc.collect()
    print("\nAll GPU memory cache cleared successfully!")

# ===================== Video Loading Function (original code unchanged) =====================
def load_video(video_data, strategy='chat'):
    bridge.set_bridge('torch')
    mp4_stream = video_data
    num_frames = 24
    decord_vr = VideoReader(io.BytesIO(mp4_stream), ctx=cpu(0))

    frame_id_list = None
    total_frames = len(decord_vr)
    if strategy == 'base':
        clip_end_sec = 60
        clip_start_sec = 0
        start_frame = int(clip_start_sec * decord_vr.get_avg_fps())
        end_frame = min(total_frames, int(clip_end_sec * decord_vr.get_avg_fps())) if clip_end_sec is not None else total_frames
        frame_id_list = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
    elif strategy == 'chat':
        timestamps = decord_vr.get_frame_timestamp(np.arange(total_frames))
        timestamps = [i[0] for i in timestamps]
        max_second = round(max(timestamps)) + 1
        frame_id_list = []
        for second in range(max_second):
            closest_num = min(timestamps, key=lambda x: abs(x - second))
            index = timestamps.index(closest_num)
            frame_id_list.append(index)
            if len(frame_id_list) >= num_frames:
                break
    video_data = decord_vr.get_batch(frame_id_list)
    video_data = video_data.permute(3, 0, 1, 2)
    return video_data

# ===================== Inference Function (original code unchanged) =====================
def inference(video_path, query, temperature=0.1):
    video_data = open(video_path, 'rb').read()
    strategy = 'chat'
    video = load_video(video_data, strategy=strategy)
    
    history = []
    inputs = model.build_conversation_input_ids(
        tokenizer=tokenizer,
        query=query,
        images=[video],
        history=history,
        template_version=strategy
    )
    inputs = {
        'input_ids': inputs['input_ids'].unsqueeze(0).to(DEVICE),
        'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(DEVICE),
        'attention_mask': inputs['attention_mask'].unsqueeze(0).to(DEVICE),
        'images': [[inputs['images'][0].to(DEVICE).to(TORCH_TYPE)]],
    }
    gen_kwargs = {
        "max_new_tokens": 2048,
        "pad_token_id": 128002,
        "top_k": 1,
        "do_sample": False,
        "top_p": 0.1,
        "temperature": temperature,
    }
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()

# ===================== Core: Dimension Scoring Function =====================
def score_by_dimension(video_path, prompt) -> dict:
    # Replace prompt placeholder
    queries = [q.replace('[[prompt]]', prompt) for q in questions]
    answers = []
    
    # Batch execute QA
    for query in tqdm(queries, desc='Evaluating video dimensions'):
        ans = inference(video_path, query)
        answers.append(1 if 'yes' in ans else -1)
    
    answers = np.array(answers, dtype=np.float32)
    dimension_scores = {}
    
    # Calculate scores for four target dimensions
    for dim_name, idx_list in DIMENSION_INDEX_MAP.items():
        dim_score = np.mean(answers[idx_list] * weight[idx_list]).item()
        # Normalize to 0~1 range for readability
        dimension_scores[dim_name] = round((dim_score + 1) / 2, 4)
    
    return dimension_scores

# ===================== Video Comparison Function (original code unchanged) =====================
def compare_two_videos(video_path1, video_path2, prompt) -> bool:
    queries = [q.replace('[[prompt]]', prompt) for q in questions]
    answers1, answers2 = [], []
    
    for query in tqdm(queries, desc='Scoring Video 1'):
        answers1.append(1 if 'yes' in inference(video_path1, query) else -1)
    for query in tqdm(queries, desc='Scoring Video 2'):
        answers2.append(1 if 'yes' in inference(video_path2, query) else -1)
    
    answers1 = np.array(answers1)
    answers2 = np.array(answers2)
    diff = answers1 - answers2
    return np.sum(diff * weight).item() > 0

# ===================== Main Function (fixed parameter order bug + added memory cleanup) =====================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VisionReward Video Evaluator - 4 Dimensions Score")
    parser.add_argument('--quant', type=int, choices=[4, 8], default=0, help='4/8 bit quantization')
    parser.add_argument('--question', type=str, default='Is there a man in the video?')
    parser.add_argument('--score', action='store_true', help='Output 4 dimension scores')
    parser.add_argument('--compare', action='store_true', help='Compare two videos')
    parser.add_argument('--video_path', type=str, required=True, help='Path to your video')
    parser.add_argument('--video_path2', type=str, default='', help='Second video for comparison')
    parser.add_argument('--prompt', type=str, required=True, help='Prompt for evaluation')
    
    # Parse arguments
    args = parser.parse_args()

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model_kwargs = {
        "torch_dtype": TORCH_TYPE,
        "trust_remote_code": True,
        "device_map": "auto"
    }
    if args.quant == 4:
        model_kwargs["load_in_4bit"] = True
    elif args.quant == 8:
        model_kwargs["load_in_8bit"] = True

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **model_kwargs).eval()

    # Execute scoring/comparison/QA + clean up GPU memory after execution
    try:
        if args.score:
            print("\n===== VisionReward 4-Dimension Score Results =====")
            scores = score_by_dimension(args.video_path, args.prompt)
            scores["video_path"] = args.video_path
            print(f"Video Path: {scores['video_path']}")
            print(f"Alignment Score: {scores['Alignment']}")
            print(f"Stability Score: {scores['Stability']}")
            print(f"Preservation Score: {scores['Preservation']}")
            print(f"Physics Plausibility Score: {scores['Physics']}")
            
            # ===================== Core Modification: Append results to JSON file =====================
            result_file = "dimension_result.json"
            # 1. Check if file exists, create empty list if not, read existing content if yes
            if os.path.exists(result_file):
                with open(result_file, 'r', encoding='utf-8') as f:
                    result_list = json.load(f)
            else:
                result_list = []
            # 2. Append new result to the list
            result_list.append(scores)
            # 3. Write to file (retain all historical data)
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(result_list, f, indent=2, ensure_ascii=False)
            print("Score appended and saved to dimension_result.json")

        elif args.compare:
            res = compare_two_videos(args.video_path, args.video_path2, args.prompt)
            print(f"\nVideo 1 is better than Video 2: {res}\n")

        else:
            ans = inference(args.video_path, args.question)
            print(f"\nAnswer: {ans}\n")
    finally:
        # Execute memory cleanup regardless of normal exit/exception
        clear_gpu_memory()