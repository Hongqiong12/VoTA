
# Towards Visualization-of-Thought Jailbreak Attack against Large Visual Language Models (NeurIPS 2025)

This repository contains the official implementation of the paper "Towards Visualization-of-Thought Jailbreak Attack against Large Visual Language Models" accepted at NeurIPS 2025.

📄 **Paper:** [Towards Visualization-of-Thought Jailbreak Attack against Large Visual Language Models](https://openreview.net/pdf/98bc54bf489ff48cae9cd4c5cff483480722ea9c.pdf)


# 🔧 Get Our Attack Data
the detail data construction pipline as shown in the picture
![主实验的结果数据](src/fig/overview.png )

you can directly get our data from  [attack_data](src/data/attack_data)

# Evaluate Result
We employed two evaluation methods: `gptfuzzer` and `gpt4-o`. For the assessment process, we utilized `common_prompt.get_eval_prompt`, the main result as show in the below

![主实验的结果数据](src/fig/main_result.png )
![防御实验的结果数据](src/fig/defend_result.png )


## ✨ Examples of Successful VoTA Attacks!
![攻击成功的case](src/fig/atatck_case1.png)
![攻击成功的case](src/fig/attack_case2.png)
![攻击成功的case](src/fig/atatck_case3.png)

# Citation
If you find this work useful, please cite our paper:
```
@inproceedings{zhongtowards,
    title={Towards Visualization-of-Thought Jailbreak Attack against Large Visual Language Models},
    author={Zhong, Hongqiong and Teng, Qingyang and Zheng, Baolin and Chen, Guanlin and Tan, Yingshui and Liu, Zhendong and Liu, Jiaheng and Su, Wenbo and Zhu, Xiaoyong and Zheng, Bo and others},
    booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems}
}
```
