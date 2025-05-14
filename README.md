
# Overview

We propose VoTA, a novel attack framework that exploits the tension between logical reasoning and safety objectives in VLMs by generating chains of images with risky visual thoughts, achieving significantly higher success rates than existing methods.


# Get Our Attack Data Synthesis Process

# Method1

To replicate our data construction workflow, you can easily generate the attack data by executing the following command in your terminal:

```bash
python3 ./src/python/produce_attack_data.py
```

# Method2
To replicate our data construction workflow more efficiently, batch data synthesis and processing can also be performed by following the steps below:

### Step 1: Synthesize risk scenarios

First, synthesize risk scenarios for all categories. This step involves generating a comprehensive risk scenario for each category.

### Step 2: Generate subprocess prompts

Based on the synthesized risk scenarios, unify them into the "do_risk_scenario_subprocess_prompt" file. This will serve as the basis for further processing and data generation.

### Step 3: Batch image synthesis

Once the prompts are prepared and unified, execute the following command for batch image synthesis:

```bash
python3 ./src/python/t2i_batch.py
```

# Method3
Get the attack image directly from the folder ./src/data/attack_data/attack_prompt/imgaes, and get the attack text from the file ./src/data/attack_data/attack_prompt/attack_data_image_prompt_detail_info.json

# Evaluate
We employed two evaluation methods: `gptfuzzer` and `gpt4-o`. For the assessment process, we utilized `common_prompt.get_eval_prompt`

## Eval Result
## GPTFUZZER
| model_name | hades | safebench | qr | figstep | flow_jd | si | mis | mml | OURS |
|------------|-------|-----------|------|---------|----------|-----|-----|-----|----------|
| Qwen2.5-VL-72B | 13.17 | 18.17 | 15.42 | 52.53 | 59.29 | 33.69 | 73.17 | 67.43 | **99.19** |
| Qwen2.5-VL-7B | 54.49 | 52.22 | 34.76 | 61.2 | 60 | 52.44 | 76.5 | 57.88 | **93.68** |
| Qwen2-VL-72B | 22.7 | 20.04 | 18.87 | 45.6 | 50.71 | 34.82 | 66.78 | 71.4 | **97.42** |
| Qwen2-VL-7B | 14.1 | 15.57 | 13.15 | 42.4 | 53.57 | 36.96 | 66.96 | 59.51 | **94.63** |
| InternVL2-8B | 17.52 | 16.96 | 16.55 | 55.74 | 54.29 | 30.18 | 44.9 | 43.52 | **90.63** |
| InternVL2-40B | 16.24 | 15.65 | 16.61 | 55.6 | 56.43 | 40.77 | 62.67 | 72.6 | **92.05** |
| MiniCPM-V2.6 | 62.72 | 42.39 | 36.01 | 26.25 | 43.57 | 35.82 | 77.37 | 53.86 | **83.79** |
| GLM-4V-9B | 24.6 | 23.43 | 17.74 | 36.6 | 45.71 | 42.49 | \N | 31.2 | **74.53** |
| LLAVA-V1.5-13B | 56.65 | 39.96 | 33.15 | 52.45 | 40 | 10.42 | 80.66 | 70.44 | **84.56** |
|------------|-------|-----------|------|---------|----------|-----|-----|-----|----------|
| AVG (Open-Source) | 31.35 | 27.15 | 22.47 | 47.60 | 51.51 | 35.29 | 68.63 | 58.65 | **90.05** |
|------------|-------|-----------|------|---------|----------|-----|-----|-----|----------|
| GPT-4o | 7.3 | 13.18 | 11.44 | 13.86 | 65.71 | 31.58 | 62.2 | 74.45 | **99.21** |
| GPT-4o-Mini | 3.63 | 8.17 | 9.3 | 19.8 | 51.43 | 28.71 | 47.78 | 71.71 | **96.63** |
| Gemini-2.0-Flash | 7.09 | 14.92 | 13.17 | 56 | 73.57 | 20.18 | 52.47 | 83.01 | **99.95** |
| Gemini-1.5-Pro | 12.78 | 23.87 | 13.95 | 52.2 | 67.14 | 40.5 | 58.06 | 83.19 | **100.00** |
| Gemini-2.5-Pro | 4.34 | 16.53 | 9.06 | 16.43 | 68.57 | 33.45 | 50.88 | 71.89 | **98.74** |
| Claude-3.5-sonnet2 | 0.16 | 1.35 | 0.54 | 7.6 | 21.43 | 11 | 5.39 | 43.44 | **51.16** |
|------------|-------|-----------|------|---------|----------|-----|-----|-----|----------|
| AVG (Commercial) | 5.88 | 13.00 | 9.58 | 27.65 | 57.98 | 27.57 | 46.13 | 71.28 | **90.95** |
|------------|-------|-----------|------|---------|----------|-----|-----|-----|----------|
| avg | 21.80 | 21.85 | 17.64 | 40.12 | 53.93 | 32.39 | 59.63 | 63.39 | **90.39** |

### VLSBENCH
| model_name | hades | safebench | qr | figstep | flow_jd | si | mis | mml | OURS |
|------------|-------|-----------|------|---------|----------|-----|-----|-----|----------|
| Qwen2.5-VL-72B | 18.15 | 33.84 | 52.68 | 84.21 | 88.57 | 68.48 | 84.78 | 90.99 | **99.73** |
| Qwen2.5-VL-7B | 58.84 | 60.05 | 72.42 | 93.19 | 88.57 | 74.64 | 92.24 | 97.27 | **99.47** |
| Qwen2-VL-72B | 34.44 | 42.77 | 61.63 | 74.2 | 95 | 78.09 | 95.79 | 92.63 | **99.68** |
| Qwen2-VL-7B | 27.63 | 39.72 | 53.54 | 77.15 | 92.86 | 78.01 | 90.19 | 89.9 | **99.74** |
| InternVL2-8B | 30.49 | 38.37 | 60.37 | 93.44 | 91.43 | 83.84 | 87.7 | 95.86 | **99.05** |
| InternVL2-40B | 23.61 | 30.15 | 54.24 | 91.4 | 91.43 | 85.09 | 87.16 | 93.25 | **98.26** |
| MiniCPM-V2.6 | 73.19 | 62.35 | 73.36 | 53.91 | 67.63 | \N | 96.84 | 84.66 | **99.58** |
| GLM-4V-9B | 33.33 | 42.32 | 54.35 | 64.53 | 73.57 | \N | \N | 53.27 | **92.32** |
| LLAVA-V1.5-13B | 65.06 | 59.78 | 73.35 | 92.45 | 95.71 | 85.14 | 97.85 | 96.46 | **98.32** |
|------------|-------|-----------|------|---------|----------|-----|-----|-----|----------|
| AVG (Open-Source) | 40.53 | 45.48 | 61.77 | 80.50 | 87.20 | 79.04 | 91.57 | 88.25 | **98.46** |
|------------|-------|-----------|------|---------|----------|-----|-----|-----|----------|
| GPT-4o | 10.44 | 26.1 | 46.03 | 40.85 | 77.14 | 72.65 | 76.85 | 87.02 | **99.74** |
| GPT-4o-Mini | 6.79 | 22.72 | 46.77 | 52.6 | 88.57 | 72.12 | 66.78 | 86.54 | **99.00** |
| Gemini-2.0-Flash | 10.55 | 27.51 | 43.06 | 84.94 | 82.73 | 75.12 | 63.1 | 91.24 | **100.00** |
| Gemini-1.5-Pro | 11.61 | 29.52 | 37.4 | 73.6 | 76.43 | 71.62 | 58 | 89.51 | **99.52** |
| Gemini-2.5-Pro | 5.99 | 24.68 | 38.89 | 51.62 | 80 | 73.72 | 59.05 | 83.28 | **99.84** |
| Claude-3.5-sonnet2 | 1.8 | 5.89 | 23.55 | 30.92 | 50.71 | 42.69 | 10.81 | 51.45 | **52.68** |
|------------|-------|-----------|------|---------|----------|-----|-----|-----|----------|
| AVG (Commercial) | 7.86 | 22.74 | 39.28 | 55.76 | 75.93 | 67.99 | 55.77 | 81.51 | **91.80** |
|------------|-------|-----------|------|---------|----------|-----|-----|-----|----------|
| avg | 28.28 | 36.95 | 53.34 | 71.22 | 82.97 | 74.30 | 77.25 | 85.72 | **95.96** |
|------------|-------|-----------|------|---------|----------|-----|-----|-----|----------|
