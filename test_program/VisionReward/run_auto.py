import subprocess
import os
import time

# ===================== Core Configuration =====================
CONDA_ENV = "visionreward"          # Conda environment name
WORK_DIR = r"D:\VisionReward"       # Working directory
SCRIPT_NAME = "inference-video.py"  # Original script file name
QUANT = 4                           # Quantization parameter (4/8/0)

# ===================== Batch Tasks: Video Path + Prompt =====================
TASKS = [
    #(r"D:\consisid\ConsisID\result5\old\0010.mp4", "Multicolored spinning 3 dimensional circles with binary codes."
    # ),
]

# ===================== Batch Execution Function =====================
def run_batch_evaluation():
    os.chdir(WORK_DIR)
    
    for idx, (video_path, prompt) in enumerate(TASKS, 1):
        print(f"\n{'='*50}")
        print(f"Task {idx}/{len(TASKS)}")
        print(f"Video: {video_path}")
        print(f"Prompt: {prompt}")
        print('='*50)

        # Build execution command
        cmd = [
            "python", SCRIPT_NAME,
            "--score",
            "--video_path", video_path,
            "--prompt", prompt,
        ]

        try:
            # Execute the script
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            print(f"Task failed: {video_path}")
        
        time.sleep(1)

    print("\nAll batch tasks executed successfully!")

if __name__ == "__main__":
    run_batch_evaluation()