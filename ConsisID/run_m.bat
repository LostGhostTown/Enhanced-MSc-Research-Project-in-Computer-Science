@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul

:: ====================== 【用户配置区 - 仅修改这里即可】 ======================
:: Conda环境名称
set CONDA_ENV=consisid_new
:: 固定的图片路径（唯一图片，修改为你的实际路径）
set IMAGE_PATH="ConsisID-data\eval\face_images\1_stars_woman_Taylor_Swift_1.png"
:: 提示词TXT文件路径（每行一个prompt，修改为你的txt路径）
set PROMPT_TXT="prompts_n.txt"
:: Python脚本/模型/输出路径（保持默认即可）
set PYTHON_SCRIPT=tools/cache_inference/teacache_inference_consisid.py
set CKPTS_PATH=ckpts
set OUTPUT_PATH=resultnew
:: 推理参数（保持默认即可）
set REL_L1_THRESH=0.1
set NUM_STEPS=50
:: ==========================================================================

:: 激活Conda环境
call conda activate %CONDA_ENV%
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

:: 检查TXT文件是否存在
if not exist %PROMPT_TXT% (
    echo 错误：未找到提示词文件 %PROMPT_TXT%！
    pause
    exit /b 1
)

:: 检查图片文件是否存在
if not exist %IMAGE_PATH% (
    echo 错误：未找到图片文件 %IMAGE_PATH%！
    pause
    exit /b 1
)

echo ==============================================
echo  配置加载完成
echo  固定图片：%IMAGE_PATH%
echo  提示词文件：%PROMPT_TXT%
echo  开始逐行执行推理任务...
echo ==============================================
echo.

:: 初始化运行计数器
set RUN_COUNT=0

:: 核心：遍历TXT文件每一行（支持空格、特殊字符、长文本）
for /f "usebackq delims=" %%a in (%PROMPT_TXT%) do (
    set "line=%%a"
    set "line=!line:"=''!"
    set /a RUN_COUNT+=1
    
    echo ===== RUN !RUN_COUNT! =====
    
    python %PYTHON_SCRIPT% ^
    --rel_l1_thresh %REL_L1_THRESH% ^
    --ckpts_path %CKPTS_PATH% ^
    --image %IMAGE_PATH% ^
    --prompt "!line!" ^
    --output_path %OUTPUT_PATH% ^
    --num_infer_steps %NUM_STEPS%

    :: 错误检测：运行失败则暂停退出
    if errorlevel 1 (
        echo 错误：第 !RUN_COUNT! 次运行执行失败！
        pause
        exit /b 1
    )
    echo.
)

:: 全部执行完成
echo ==============================================
echo  全部任务执行完成！总运行次数：!RUN_COUNT!
echo ==============================================

pause