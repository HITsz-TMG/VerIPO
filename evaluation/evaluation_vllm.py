import os
os.environ["DECORD_EOF_RETRY_MAX"] = "40960"
import sys
import json
import argparse
import torch
import re
from tqdm import tqdm
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

OPTION_QUESTION_TEMPLATE = "You should first thinks about the reasoning process in the mind and then provides the user with the answer. Your answer must be in latex format and wrapped in $...$.The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> Since ...., so the answer is B. </think><answer> $B$ </answer>, which means your output should start with <think> and end with </answer>.\nQuestion:\n{Question}"

NUMBER_QUESTION_TEMPLATE = "You should first thinks about the reasoning process in the mind and then provides the user with the answer. Your answer must be in latex format and wrapped in $...$.The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> Since ...., so the answer is $2$. </think><answer> $2$ </answer>, which means your output should start with <think> and end with </answer>.\nQuestion:\n{Question}\nYou must provide the answer in the <answer> </answer> tag, and the answer must be a number."


def eval_model(args):
    # 设置采样参数
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens, 
        temperature=args.temperature, 
        top_p=args.top_p, 
        repetition_penalty=args.repetition_penalty,
    )
    
    # 创建输出目录
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # 加载模型和处理器
    model = LLM(
        model=args.model_path,
        dtype=torch.bfloat16,
        gpu_memory_utilization=0.5,
        tensor_parallel_size=args.tensor_parallel_size
    )
    
    processor = AutoProcessor.from_pretrained(args.model_path)
    
    # 加载数据
    data = json.load(open(args.prompt_path, "r"))
    # data = data[199:]
    # 准备所有消息
    # all_llm_inputs = []
    all_outputs = []
    for item in tqdm(data):
        p_question = item["processed_question"]
        p_question = p_question.replace("\nAnswer with the option's letter from the given choices directly.", "")
        # 选项类问题
        if item["answer"].isalpha():
            p_question = OPTION_QUESTION_TEMPLATE.format(Question=p_question)
        # 数字类问题
        else:
            p_question = NUMBER_QUESTION_TEMPLATE.format(Question=p_question)
        # 其他问题
        
        print(p_question)
        # 构建消息
        if "image" not in item:
            message = [{
                "role": "user",
                "content": [
                    {
                        "type": "video", 
                        "video": os.path.join(args.video_dir ,item["video_path"]), 
                        "max_pixels": args.video_max_pixels,
                        "max_frames": args.video_max_frames,
                        "fps": args.video_fps
                    },
                    {
                        "type": "text",
                        "text": p_question
                    }
                ]
            }]
        else:
            image = item["image"]
            base64_qwen = f"data:image;base64,{image}"
            message = [{
                "role": "user",
                "content": [
                    {
                        "type": "video", 
                        "video": os.path.join(args.video_dir ,item["video_path"]), 
                        "max_pixels": args.video_max_pixels,
                        "max_frames": args.video_max_frames,
                        "fps": args.video_fps
                    },
                    {
                        "type": "image_url", 
                        "image_url": base64_qwen
                    },
                    {
                        "type": "text",
                        "text": p_question
                    }
                ]
            }]

        # 转换为vLLM输入格式
        prompt = processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs, video_kwargs = process_vision_info(message, return_video_kwargs=True)
        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        
        if video_inputs is not None:
            mm_data["video"] = video_inputs
        
        llm_input = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": video_kwargs,
        }
        
        outputs = model.generate([llm_input], sampling_params=sampling_params, use_tqdm=False)
        generated_text = outputs[0].outputs[0].text
        print(generated_text)
        all_outputs.append(generated_text)
        
    # 评估结果
    final_output = []
    
    for input_example, model_output in zip(data, all_outputs):
        ground_truth = input_example['answer']
        
        result = {
            'question': input_example,
            'ground_truth': ground_truth,
            'model_output': model_output,
        }
        final_output.append(result)
        
    
    # 保存结果
    with open(args.output_path, "w") as f:
        json.dump(final_output, f, indent=2)
    
    print(f"Results saved to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--prompt_path", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1e-6)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--video_fps", type=float, default=2.0)
    parser.add_argument("--video_max_pixels", type=int, default=200704)
    parser.add_argument("--video_max_frames", type=int, default=128)
    parser.add_argument("--video_dir", type=str, default="")
    parser.add_argument("--repetition_penalty", type=int, default=1.05)
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size")
    args = parser.parse_args()
    
    eval_model(args)