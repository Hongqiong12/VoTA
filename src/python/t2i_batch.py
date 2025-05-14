# -*- coding: utf-8 -*-            
# @Time : 2025/1/16 14:29
# @Autor: joanzhong
# @FileName: t2i_batch.py 文生图批量合成
# @Software: IntelliJ IDEA
import argparse
import json
import logging
import os
import time
import torch

import torch.multiprocessing as mp

from functools import partial
from typing import Union, List, Callable, Iterator
from diffusers import StableDiffusion3Pipeline
from torch.utils.data import DataLoader
from src.t3mio.base import FiniteStreamDataset


logger = logging.getLogger(__name__)

if os.environ.get("LOCAL_PROCESS_RANK"):
    is_local_first_process = int(os.environ.get("LOCAL_PROCESS_RANK", "0")) == 0
else:
    is_local_first_process = int(os.environ.get("LOCAL_RANK", "0")) == 0
is_global_first_process = int(os.environ.get("RANK", "0")) == 0


class Timer(object):
    def __init__(self) -> None:
        self.timestep = None

    def start(self):
        self.timestep = time.time()
        return self

    def elapsed(self):
        cur_time = time.time()
        return cur_time - self.timestep


def set_logger_level(rank):
    logging.basicConfig(format=f"%(asctime)s [%(levelname)s][%(filename)s:%(lineno)d][Rank {rank}] %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)


def get_time(s: int):
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{int(h)}:{int(m)}:{int(s)}"


def get_local_rank():
    return int(os.environ.get('LOCAL_RANK', 0))


def get_global_rank():
    return int(os.environ.get('RANK', 0))


def get_world_size():
    return int(os.environ.get('WORLD_SIZE', 0))


def get_sd_image(pipe, prompt):
    logger.info(f"=====文生图的prompt：{prompt}====")
    image = pipe(
        prompt,
        negative_prompt="",  # 负面提示词，让模型避免生成negative prompt所表述的概念
        num_inference_steps=60,  # 一般来说，使用的步数越多，结果越好
        guidance_scale=7.5,  # 调整它可以更好的使用图像质量更好或更具备多样性。值介于7和8.5之间通常是稳定扩散的好选择。默认情况下，管道使用的guidance_scale为7.5
        max_sequence_length=512
    ).images[0]
    return image


def collate(batch):
    id_arr = []
    prompt_arr = []
    oss_path_arr = []
    for id, prompt, oss_path in batch:
        id_arr.append(id)
        prompt_arr.append(prompt)
        oss_path_arr.append(oss_path)
    return id_arr, prompt_arr, oss_path_arr


class JSONLDataset(FiniteStreamDataset):
    """Reads data from JSONL files.

    Each line of the input files should be a valid JSON object.
    """

    def __init__(self, jsonl_filepaths: Union[str, List[str]], consumed_samples: int = 0,
                 map_func: Callable[[dict], dict] = None):
        """
        Args:
            jsonl_filepaths: Path to the JSONL file(s). Can be a string or a list of strings.
            consumed_samples: Number of samples already consumed. Used for resuming training.
            map_func: A function to apply to each JSON object after reading.
        """

        super().__init__()
        if isinstance(jsonl_filepaths, str):
            jsonl_filepaths = [jsonl_filepaths]

        self.jsonl_filepaths = jsonl_filepaths
        self.consumed_samples = consumed_samples
        self._map_func = map_func
        self.row_count_total = 0
        self.start_pos = 0
        self.end_pos = 0
        self.row_count = 0
        self.intra_data_offset = 0
        self.is_initialized = False

    def initialize(self, slice_id: int, slice_count: int):
        """Initializes the dataset after determining data splits.
        :param slice_id: Current slice ID for distributed training.
        :param slice_count: Total number of slices for distributed training.
        """
        all_rows = []
        for filepath in self.jsonl_filepaths:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    all_rows.append(json.loads(line))

        self.row_count_total = len(all_rows)
        self.start_pos, self.end_pos = self._get_slice_range(self.row_count_total, slice_id, slice_count)
        self.row_count = self.end_pos - self.start_pos

        if self.consumed_samples > 0:
            self.intra_data_offset = self._set_data_offset(self.consumed_samples)

        self.is_initialized = True

    def _set_data_offset(self, consumed_samples: int):
        """Sets the intra-data offset for resuming training."""

        consumed_samples_per_worker = consumed_samples // self.slice_count
        return consumed_samples_per_worker

    def _get_slice_range(self, row_count: int, worker_id: int, num_workers: int):
        """Calculates the start and end positions for a given worker."""

        size = row_count // num_workers
        remainder = row_count % num_workers
        start = worker_id * size + min(worker_id, remainder)
        end = (worker_id + 1) * size + min(worker_id + 1, remainder)
        return start, end

    @property
    def num_samples_local(self) -> int:
        return self.row_count

    @property
    def num_samples_global(self) -> int:
        return self.row_count_total

    def build_data_stream(self, worker_id: int, num_workers: int) -> Iterator[dict]:
        """Builds the data stream for a given worker."""

        assert self.is_initialized

        start, end = self._get_slice_range(self.row_count, worker_id, num_workers)
        start += self.intra_data_offset // num_workers
        global_start = self.start_pos + start
        global_end = self.start_pos + end

        for idx in range(global_start, min(global_end, self.end_pos)):
            data = self._map_func(self.get_item(idx)) if self._map_func else self.get_item(idx)
            yield data

    def get_item(self, idx: int) -> dict:
        """Get the item at the specified index within the dataset's file list"""
        cumulative_lines = 0
        for filepath in self.jsonl_filepaths:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if cumulative_lines + line_num == idx:
                        return json.loads(line)

                cumulative_lines += line_num + 1  # Add 1 to account for 0-indexing

        raise IndexError(f"Index {idx} is out of range")


def preprocess(line):
    id = line["id"]
    prompt = line["prompt"]
    oss_path = line["oss_path"]
    return id, prompt, oss_path


def save_image_with_dir_creation(image, filepath):
    """Saves a PIL Image, creating necessary directories if they don't exist."""
    try:
        # Extract directory path
        directory = os.path.dirname(filepath)

        # Create directory if it doesn't exist
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        # Save the image
        image.save(filepath)
        print(f"Image saved to: {filepath}")

    except Exception as e:
        print(f"Error saving image: {e}")


def main(arguments):
    # 参数解析
    for k, v in arguments.__dict__.items():
        print(k, v)

    set_logger_level(get_local_rank())
    # print('args:\n{}\n'.format(args))
    torch.cuda.set_device(get_local_rank())

    # 加载json文件
    dataset = JSONLDataset(args.input_data_path)
    dataset.initialize(slice_id=get_global_rank(), slice_count=get_world_size())
    dataset.map(preprocess)

    #
    logger.info('is_local_first_process:{}'.format(is_local_first_process))
    collate_fn = partial(collate)
    dataloader = DataLoader(dataset, batch_size=arguments.per_device_batch_size, shuffle=False,
                            num_workers=arguments.num_dataloader_workers, pin_memory=True, collate_fn=collate_fn)
    # 加载模型
    sd_model_path = arguments.sd_model_path
    pipe = StableDiffusion3Pipeline.from_pretrained(sd_model_path, torch_dtype=torch.bfloat16)
    if torch.cuda.is_available():
        logger.info("======将pipeline移动到GPU=========== ")
        pipe.to("cuda")  # 将pipeline移动到GPU
    # 单机运算逻辑
    for id_arr, prompt_arr, oss_path_arr in dataloader:
        #
        for i in range(len(id_arr)):
            oss_path = oss_path_arr[i]
            save_path = f"/data/oss_bucket_0/{oss_path}"
            # if not os.path.exists(save_path):
            #     image_id = id_arr[i]
            #     # 先合成stable difussion的数据
            #     sd_prompt = prompt_arr[i]
            #     image_sd = get_sd_image(pipe, sd_prompt)
            #     logger.info(f"==========数据ID：{image_id}, 存储地址：{save_path}==========")
            #     save_image_with_dir_creation(image_sd, save_path)
            # 如果是需要重新覆盖数据的话
            image_id = id_arr[i]
            # 先合成stable difussion的数据
            sd_prompt = prompt_arr[i]
            image_sd = get_sd_image(pipe, sd_prompt)
            logger.info(f"==========数据ID：{image_id}, 存储地址：{save_path}==========")
            save_image_with_dir_creation(image_sd, save_path)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--used_columns", default=None, type=str)
    parser.add_argument("--log_interval", default=5, type=int)
    # table args
    parser.add_argument('--input_data_path', default="", type=str)
    # SD的模型地址
    parser.add_argument('--sd_model_path', default="", type=str)
    # 字体
    parser.add_argument('--front_path', default="", type=str)

    #
    parser.add_argument("--per_device_batch_size", default=8, type=int)
    parser.add_argument("--num_dataloader_workers", default=4, type=int)

    return parser.parse_args()


if __name__ == "__main__":
    node_rank = int(os.environ['RANK'])
    node_world_size = int(os.environ['WORLD_SIZE'])
    print('world_size', node_world_size, 'rank', node_rank)

    gpu_per_node = torch.cuda.device_count()

    print('gpu_per_node:', gpu_per_node)
    if is_global_first_process:
        mp.set_start_method("spawn")

    args = get_arguments()

    main(args)
