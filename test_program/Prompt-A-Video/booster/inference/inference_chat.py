import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ================= 官方原版 TASK_PROMPT，一字未改 =================
TASK_PROMPT = """You need to refine user's input prompt. The user's input prompt is used for video generation task. You need to refine the user's prompt to make it more suitable for the task.
You will be prompted by people looking to create detailed, amazing videos. The way to accomplish this is to take their short prompts and make them extremely detailed and descriptive. You will only ever output a single video description per user request.
You should refactor the entire description to integrate the suggestions. Original prompt:\n"""

def main(
    model_name,
    input_file: str=None,
    input_prompt: str=None,
    output_file: str=None,
    max_new_tokens =256,
    seed: int=42,
    do_sample: bool=True,
    top_p: float=1.0,
    temperature: float=1.0,
    top_k: int=50,
    repetition_penalty: float=1.0,
    **kwargs
):
    res = []
    # 读取输入提示词
    if input_file:
        with open(input_file, 'r', encoding='utf-8') as f:
            prompts_src = f.readlines()
    elif input_prompt:
        prompts_src = [input_prompt]
    else:
        sys.exit('input is None.')
    
    prompts_src = [p.strip() for p in prompts_src]

    # 设置随机种子，保证结果可复现
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

    print(f"正在加载完整模型: {model_name}")
    # 加载tokenizer和完整的提示词优化模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    model.eval()
    print("模型加载完成！开始优化提示词...")

    # 循环优化每个提示词
    for user_prompt in prompts_src:
        print(f"----------------\n原始提示词:\n{user_prompt}")

        # 完全对齐官方原版的对话格式
        full_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{TASK_PROMPT}{user_prompt}\nNew prompt:\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

        # 生成优化后的提示词，参数和官方完全一致
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                use_cache=True,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.eos_token_id
            )

            # 解析输出，和官方逻辑完全一致
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = output_text.split('assistant')[-1].strip()

            print(f"优化后提示词:\n{response}")
            res.append([user_prompt, response])
    
    # 保存结果到json文件
    print(f"\n✅ 全部优化完成，正在保存结果到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=4)

if __name__ == "__main__":
    # 极简参数解析，兼容你之前的运行命令
    args = {}
    for i in range(1, len(sys.argv), 2):
        key = sys.argv[i].lstrip('--')
        val = sys.argv[i+1]
        # 类型自动转换
        if val.lower() == 'true': val = True
        elif val.lower() == 'false': val = False
        elif val.isdigit(): val = int(val)
        else:
            try: val = float(val)
            except: pass
        args[key] = val
    
    main(**args)