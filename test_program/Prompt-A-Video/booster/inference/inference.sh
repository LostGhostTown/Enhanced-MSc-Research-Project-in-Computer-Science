CUDA_VISIBLE_DEVICES=0 \
python inference_chat.py --model_name ../../Prompt_A_Video_CV \
    --input_file ../../input.txt --output_file ../../output.json
# python inference_chat.py --model_name /path/to/base_model --peft_model /path/to/lora \
#     --input_file vbench_test.txt --output_file ./example_res.json
read -p "按任意键退出..."