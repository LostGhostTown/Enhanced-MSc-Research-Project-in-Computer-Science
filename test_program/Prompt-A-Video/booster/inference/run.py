import subprocess
import sys

# 直接调用 inference_chat.py，传入参数
cmd = [
    sys.executable, "inference_chat.py",
    "--model_name", "../../Prompt_A_Video_CV",
    "--input_file", "../../input.txt",
    "--output_file", "../../output.json"
]

print("正在运行提示词优化...")
print(f"命令: {' '.join(cmd)}")
print("-" * 50)

# 运行并实时显示输出
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='gbk')
for line in process.stdout:
    print(line, end='')
process.wait()

print("-" * 50)
if process.returncode == 0:
    print("✅ 运行成功！结果在 output.json")
else:
    print(f"❌ 运行失败，错误码: {process.returncode}")

input("\n按回车键退出...")