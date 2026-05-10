import os
import base64
import dashscope

dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")  # Replace with your API Key

# ===================== 【Required Configuration】Modify Here =====================
# 2. Paths of 3 local videos (replace with your video file paths)
VIDEO_PATHS = [
    r"D:\consisid\ConsisID\finalresult\newVideo\0007.mp4",
    r"D:\consisid\ConsisID\finalresult\oldVideo\0007.mp4",
    r"D:\consisid\ConsisID\resultVPO\0006.mp4"
]
# 3. Model to use (Qwen3.5 Plus Multimodal, supports video)
MODEL = "qwen3.5-plus"
# ================================================================================

def video_to_base64(video_path: str) -> str:
    """
    Convert local video file to Base64 encoding (required format for Qianwen API)
    :param video_path: Local path of the video
    :return: Base64 encoded string
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    with open(video_path, "rb") as f:
        video_bytes = f.read()
        # Concatenate the format required by the API: data:video/mp4;base64,encoded content
        base64_str = f"data:video/mp4;base64,{base64.b64encode(video_bytes).decode('utf-8')}"
    return base64_str

def evaluate_three_videos(video_paths: list) -> str:
    """
    Call Qianwen API to evaluate and score 3 videos at once
    :param video_paths: List of 3 video paths
    :return: AI scoring result
    """
    # Step 1: Convert 3 videos to Base64
    video_base64_list = [video_to_base64(path) for path in video_paths]
    
    # Step 2: Construct Prompt (force standardized AI scoring for structured results)
    prompt = """
    A woman stands in a sunlit flower shop, arranging a bouquet of peonies. Her fingers brush off a stray petal from her apron, and she leans in to smell a pink bloom—her eyes softening as the scent lingers. A small radio on the counter plays a folk song, and water droplets from the flower vases glisten on the wooden tabletop.
    You are a professional video quality evaluator. Based on this prompt, evaluate the quality of Video 1, Video 2, and Video 3 at once. Score the three videos on a 10-point scale with 4 decimal places from the perspectives of prompt alignment and video authenticity. Output strictly in the following format:
       Video 1: Total Score X | Reason: xxx
       Video 2: Total Score X | Reason: xxx
       Video 3: Total Score X | Reason: xxx
    No extra words, output only the evaluation results
    """
    
    # Step 3: Construct API request message (3 videos + text instruction)
    messages = [
        {
            "role": "user",
            "content": [
                {"video": video_base64_list[0]},
                {"video": video_base64_list[1]},
                {"video": video_base64_list[2]},
                {"text": prompt}
            ]
        }
    ]

    # Call the official API
    response = dashscope.MultiModalConversation.call(
        api_key=DASHSCOPE_API_KEY,
        model=MODEL,
        messages=messages
    )

    # Extract results (strictly follow the official response format)
    if response.status_code == 200:
        return response.output.choices[0].message.content[0]["text"]
    else:
        return f"API call failed: {response.code} - {response.message}"

if __name__ == "__main__":
    print("===== Qwen3.5 Video Quality Batch Evaluation =====")
    print(f"Videos to evaluate: {VIDEO_PATHS}\n")
    
    try:
        result = evaluate_three_videos(VIDEO_PATHS)
        print("===== Evaluation Result =====")
        print(result)
    except Exception as e:
        print(f"Program exception: {str(e)}")