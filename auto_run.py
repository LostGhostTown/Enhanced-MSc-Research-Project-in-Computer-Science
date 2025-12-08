
import os
import sys
import subprocess
import time
import argparse
import functools
print = functools.partial(print, flush=True)
# ---------------------- configuration----------------------
PYTHON_SCRIPT = "tools/cache_inference/teacache_inference_consisid.py"  
REL_L1_THRESH = 0.1  
CKPTS_PATH = "ckpts"  
NUM_STEPS = 50 
QIANFAN_ANALYSIS_SCRIPT = "qianfan_analysis.py"  
# -----------------------------------------------------------------------------------

def get_latest_video_path(save_dir):
    """Get the path of the latest generated video"""
    os.makedirs(save_dir, exist_ok=True)
    file_count = len([f for f in os.listdir(save_dir) if os.path.isfile(os.path.join(save_dir, f))])-1
    video_path = os.path.join(save_dir, f"{file_count:04d}.mp4")
    return video_path

def run_video_generation(image_path, current_prompt, run_idx, save_path):
    """Execute the video generation command"""
    print(f"\n{'='*30} Run {run_idx} Video Generation {'='*30}")
    print(f"Current prompt for this round: {current_prompt}")
    cmd = [
        sys.executable,
        "-u",
        PYTHON_SCRIPT,
        f"--rel_l1_thresh={REL_L1_THRESH}",
        f"--ckpts_path={CKPTS_PATH}",
        f"--image={image_path}",
        f"--prompt={current_prompt}",  
        f"--output_path={save_path}",
        f"--num_infer_steps={NUM_STEPS}"
    ]
    print(f"<think></think>Execute the video generation command:{' '.join(cmd)}")
    print(f"Videos will be saved to: {save_path}")
    print(f"\n{'='*30}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    for line in process.stdout:
        print(line, end='', flush=True)
    
    process.wait()
    if process.returncode != 0:
        print(f"Error in Run {run_idx}: Video generation failed!", flush=True)
        exit(1)
        return None 
    print(f"Run {run_idx} video generation succeeded!", flush=True)
    return True

def main():
    # Parse the 4 required user inputs: image path + initial prompt + save path + loop count
    parser = argparse.ArgumentParser(description="Automated iterative video generation + Qianfan analysis (supports user-specified loop count)")
    # Required parameter 1: input image path
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Input image path (relative/absolute, e.g., 'input.jpg' or 'C:/path/to/input.jpg')"
    )
    # Required parameter 2: initial prompt (used in the 1st round)
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Initial prompt (used in the 1st round, enclose in quotes if it contains spaces, e.g., 'A woman stands in a sunlit flower shop')"
    )
    # Required parameter 3: save path (videos + TXT in the same directory)
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Save path (videos and analysis TXT will be saved here, e.g., './output' or 'C:/path/to/save')"
    )
    # Required parameter 4: loop count (user-specified, replaces fixed 3 times)
    parser.add_argument(
        "--loop_num",
        type=int,
        required=True,
        help="Number of loops (e.g., 3 means generating and analyzing videos 3 times, supports any positive integer)"
    )
    args = parser.parse_args()
    
    # Parse parameters
    image_path = args.image
    initial_prompt = args.prompt  # Initial prompt for the 1st round
    save_path = args.save_path
    loop_num = args.loop_num
    
    # Pre-check
    if not os.path.exists(image_path):
        print(f"Error: The input image address does not exist → {image_path}")
        sys.exit(1)
    if loop_num <= 0:
        print(f"Error: The loop count must be a positive integer → You entered {loop_num}")
        sys.exit(1)
    try:
        os.makedirs(save_path, exist_ok=True)
        print(f"Save path is ready: {save_path} (videos and TXT will be saved here)")
    except Exception as e:
        print(f"Error: Unable to create save path → {save_path}, error: {e}")
        sys.exit(1)

    # qianfan analysis script import
    sys.path.append(os.path.dirname(os.path.abspath(QIANFAN_ANALYSIS_SCRIPT)))
    from qianfan import analyze_video_with_qianfan

    # Initialize the current round prompt (use the user's initial prompt for the 1st round)
    current_prompt = initial_prompt

    # Loop execution: user-specified number of times
    for run_idx in range(1, loop_num + 1):
        print(f"\n{'='*60} Round {run_idx}/{loop_num} Start {'='*60}")
        print(f"Current round prompt: {current_prompt}")
        
        # 1. Generate video (using the current round prompt)
        gen_success = run_video_generation(image_path, current_prompt, run_idx, save_path)
        if not gen_success:
            print(f"{'='*60} Round {run_idx}/{loop_num} failed {'='*60}\n")
            
            exit(1)

        # 2. get_latest_video_path
        video_path = get_latest_video_path(save_path)
        time.sleep(1)
        if not os.path.exists(video_path):
            print(f"Error: The video generated in round {run_idx} does not exist: {video_path}")
            print(f"{'='*60} Round {run_idx}/{loop_num} failed {'='*60}\n")
            exit(1)
        print(f"\n{'='*60} Round {run_idx}/{loop_num} Qianfan Analysis Start {'='*60}")
        # 3. Call Qianfan analysis (pass in the current round prompt, return the next round prompt)
        next_prompt = analyze_video_with_qianfan(
            video_path=video_path,
            original_prompt=current_prompt,  
            save_path=save_path,
            maxtokens=8000
        )

        # 4. Update the prompt for the next round (if the analysis is successful)
        if next_prompt:
            current_prompt = next_prompt  
            print(f"Round {run_idx} iteration complete, next round prompt updated")
        else:
            print(f"Round {run_idx} analysis failed")
            exit(1)

        print(f"{'='*60} Round {run_idx}/{loop_num} Complete {'='*60}\n")


    print(f"\nIteration process complete!")
    print(f"Total loops: {loop_num}")
    print(f"Video: {save_path}（0001.mp4 ~ {loop_num:04d}.mp4）")
    print(f"Result: {save_path}（analysis_0001.txt ~ analysis_{loop_num:04d}.txt）")

if __name__ == "__main__":
    main()