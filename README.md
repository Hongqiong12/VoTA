
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

# evaluate
We employed two evaluation methods: `gptfuzzer` and `gpt4-o`. For the assessment process, we utilized `common_prompt.get_eval_prompt`
