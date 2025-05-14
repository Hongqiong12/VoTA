# -*- coding: utf-8 -*-            
# @Time : 2025/5/14 14:19
# @Autor: joanzhong
# @FileName: main.py
# @Software: IntelliJ IDEA
import argparse
import json

import json_repair
import torch
from diffusers import StableDiffusion3Pipeline

from src.python.src.utils.common_prompt import GetPrompt
from src.python.src.utils.constant_param import get_ori_categories
from src.python.src.utils.get_mllm_response import query_base
from src.python.src.utils.image_utils import arrange_images


def get_sd_image(pipe, prompt):
    image = pipe(
        prompt,
        negative_prompt="",  # 负面提示词，让模型避免生成negative prompt所表述的概念
        num_inference_steps=60,  # 一般来说，使用的步数越多，结果越好
        guidance_scale=7.5,  # 调整它可以更好的使用图像质量更好或更具备多样性。值介于7和8.5之间通常是稳定扩散的好选择。默认情况下，管道使用的guidance_scale为7.5
        max_sequence_length=512
    ).images[0]
    return image


def from_subprocess_get_action_entiy_prompt(subprocess_res):
    process_inference = json_repair.loads(subprocess_res)
    entity_t2i_prompt = [a["entity_t2i_prompt"] for a in process_inference if
                         a.__contains__("entity_t2i_prompt")]
    entity_name = [a["entity_name"] for a in process_inference if a.__contains__("entity_name")]
    action_name = [a["action_name"] for a in process_inference if a.__contains__("action_name")]
    return entity_t2i_prompt, entity_name, action_name


def produce_attack_data(base_save_image_path, attack_data_save_path):
    id = 0
    tot_attack_data = []
    tot_cate = get_ori_categories()
    with open(attack_data_save_path, 'a', encoding='utf-8') as json_file:
        for cate in tot_cate:
            first_cate, second_cate = cate.split("|")
            # step1:get risk scenario
            risk_scenario_prompt = prompt_module.get_risk_scenario_prompt(first_cate, second_cate)
            risk_scenario = query_base(risk_scenario_produce_model, risk_scenario_prompt, api_key, image_path=None)
            for cur_risk_scenario in risk_scenario.split("\n"):
                cur_risk_scenario = cur_risk_scenario.split(".")[1]
                # step2:from cate and risk_scenario get doing risk scenario subprocess
                do_risk_scenario_subprocess_prompt = prompt_module.get_do_risk_scenario_process(cate, cur_risk_scenario)
                subprocess = query_base(atttack_llm_model, do_risk_scenario_subprocess_prompt, api_key, image_path=None)
                # 解析结果
                entity_t2i_prompt, entity_name, action_name = from_subprocess_get_action_entiy_prompt(subprocess)
                # 合成子图
                subgraph = []
                for sd_prompt in entity_t2i_prompt:
                    subgraph.append(get_sd_image(pipe, sd_prompt))
                # 子图拼接并进行存储
                image = arrange_images(subgraph, entity_name, font_dir)
                image_path = f"{base_save_image_path}/{cate}/{id}.png"
                image.save(image_path)
                # 攻击text prompt 合成
                instruction = prompt_module.get_attack_prompt(action_name, cur_risk_scenario)
                tot_attack_data.append([id, first_cate, second_cate, instruction, image_path])
                id += 1
                # save data
                write_json = {
                    "id": id,
                    "first_cate": first_cate,
                    "second_cate": second_cate,
                    "instruction": instruction,
                    "img_path": image_path
                }
                print(write_json)
                json.dump(write_json, json_file, ensure_ascii=False)
                json_file.write('\n')


def get_arguments():
    parser = argparse.ArgumentParser()
    # SD的模型地址
    parser.add_argument('--sd_model_path', default="", type=str)
    # 字体
    parser.add_argument('--front_path', default="", type=str)
    # 数据构建过程中的几个模型
    parser.add_argument('--risk_scenario_produce_model', default="gemini-1.5-pro", type=str)
    parser.add_argument('--attack_llm_model', default="gemini-1.5-pro", type=str)
    #
    parser.add_argument('--base_save_image_path', default="./src/data/attack_data/attack_images", type=str)
    parser.add_argument('--attack_data_save_path', default="./src/data/attack_data/attack_prompt", type=str)
    #
    parser.add_argument('--api_key', default="", type=str)

    return parser.parse_args()


if __name__ == '__main__':
    # param
    arguments = get_arguments()
    api_key = arguments.api_key
    risk_scenario_produce_model = arguments.risk_scenario_produce_model
    atttack_llm_model = arguments.attack_llm_model
    base_save_image_path = arguments.base_save_image_path
    attack_data_save_path = arguments.attack_data_save_path
    font_dir = arguments.front_path

    #
    prompt_module = GetPrompt()

    # init sd model
    sd_model_path = arguments.sd_model_path
    pipe = StableDiffusion3Pipeline.from_pretrained(sd_model_path, torch_dtype=torch.bfloat16)
    if torch.cuda.is_available():
        pipe.to("cuda")

    # 合成数据
    produce_attack_data(base_save_image_path, attack_data_save_path)
