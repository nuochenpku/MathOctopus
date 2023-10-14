
# Overview

This repository contains resources for accessing the official benchmarks, codes and checkpoints of the paper: "[MathOctopus: Building Math Generalist Models through Hybrid Instruction Tuning](https://arxiv.org/pdf/2309.05653.pdf)".


This work pioneers exploring and building powerful  Multilingual Math Reasoning (xMR) LLMs. To accomplish  this, we make the following works:

- **MGSM8KInstruct**,  the first multilingual math reasoning instruction dataset,  encompassing ten distinct languages, thus addressing the issue of training data scarcity in xMR tasks.
- **MSVAMP**, an out-of-domain xMR test dataset, to conduct a more exhaustive and comprehensive evaluation of the model’s multilingual mathematical capabilities.
- **MathOctopus**, our  effective Multilingual Math Reasoning  LLMs,  training with  different strategies, which notably outperform conventional open-source LLMs and exhibit superiority over ChatGPT in few-shot scenarios.



# 🐙🐙**MathOctopus** 🐙🐙
This repo contains the code, data, and models for "[MathOctopus: Building Math Generalist Models through Hybrid Instruction Tuning](https://arxiv.org/pdf/2309.05653.pdf)"

<div align="center">
 🔥 🔥 🔥 Check out our <a href = "https://tiger-ai-lab.github.io/MathOctopus/">[Project Page]</a> for more results and analysis!
</div>

<br>
<div align="center">
  <img src="MathOctopus_github.png" width="80%" title="Introduction Figure">
</div>

## Official Website

### Datasets 


Our dataset and models are all available at Huggingface.
🦑
🤗 [MathInstruct Dataset](https://huggingface.co/datasets/TIGER-Lab/MathInstruct)


Or you can directly download them from

#  Models
|  Base Model: LLama   	| Parallel-Training                                         	| Cross-Training                                                       	|
|-----	|---------------------------------------------------------------	|---------------------------------------------------------------------------	|
| 7B-LLaMA 2  	| 🐙 [MathOctopus-Parallel-7B](https://huggingface.co/Mathoctopus/Parallel_7B)   	| 🐙 [MathOctopus-Cross-7B](https://huggingface.co/Mathoctopus/Cross_7B)  	|
|| 🐙[MathOctopus-Parallel-xRFT-7B](https://huggingface.co/TIGER-Lab/MathOctopus-7B)|🐙[MathOctopus-Cross-xRFT-7B](https://huggingface.co/TIGER-Lab/MathOctopus-7B)|
| 13B-LLaMA 2 	| 🐙 [MathOctopus-Parallel-13B](https://huggingface.co/TIGER-Lab/MathOctopus-13B) 	| 🐙 [MathOctopus-Cross-13B](https://huggingface.co/TIGER-Lab/MathOctopus-Coder-13B) 	|
| 33B-LLaMA 1 	| 🐙 [MathOctopus-Parallel-33B](https://huggingface.co/TIGER-Lab/MathOctopus-13B)                                                             	| 🐙 [MathOctopus-Cross-33B](https://huggingface.co/TIGER-Lab/MathOctopus-Coder-34B) 	|
| 70B-LLaMA 2 	| Coming soon!	| Coming Soon!                                                                         	|

## **News**

[Oct. 10] We update our decoding method to hybrid decoding: first try PoT to generate a program, if it is not excutable, we will regenerate a CoT solution as the final answer. This hybrid decoding method improves the peformance significantly. Check our updated paper Appendix for more details. 

## **Table of Contents**

- [ℹ Introduction](#introduction)
- [⚙️ Installation](#installation)
- [🛠️ Training and Inference](#training-and-inference)
- [📜 License](#license)
- [📖 Citation](#citation)

## **Introduction**
We introduce MathOctopus 🐙, a series of open-source large language models (LLMs) specifically tailored for general math problem-solving. The MathOctopus models are trained on MathInstruct, a meticulously curated instruction tuning dataset that is lightweight yet generalizable. MathInstruct is compiled from 13 math rationale datasets, six of which are newly curated by this work. It uniquely focuses on the hybrid use of chain-of-thought (CoT) and program-of-thought (PoT) rationales, and ensures extensive coverage of diverse mathematical fields. 
## **Installation**

Clone this repository and install the required packages:

```bash
git clone https://github.com/TIGER-AI-Lab/MathOctopus.git
cd MathOctopus
pip install -r requirements.txt
```

## **Training and Inference**

### **Data Loading**

Run the following command to preprocess the data:

```python
from datasets import load_dataset

dataset = load_dataset("TIGER-Lab/MathInstruct")
```

### **Quick Start**
To play with our model, run:

```python
from transformers import pipeline
pipeline = pipeline("text-generation", "TIGER-Lab/MathOctopus-Coder-7B")

alpaca_template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n### Instruction:\n{query}\n\n### Response:"

query = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"

### By default, MathOctopus will output the Chain-of-thought (CoT) rationale
rationale_prefix = ""

### You can let MathOctopus output Program-of-thought (PoT) rationale by simply adding
rationale_prefix = " Let's write a program."

input = alpaca_template.format(query = query + rationale_prefix)

output = pipeline(input)[0]['generated_text']
print(output)
```

### **Large-scale Evaluation**

To replicate the experimental results in our paper, run:

```bash
### For open-eneded questions, the dataset should be one of 
### ['gsm8k', 'svamp', 'math', 'numglue', 'deepmind', 'simuleq'] 
### We first try PoT and if the generated program is not executable, we shift to CoT

dataset='math'

python run_open.py \
  --model "TIGER-Lab/MathOctopus-Coder-7B" \
  --shots 0 \
  --stem_flan_type "pot_prompt" \
  --batch_size 8 \
  --dataset $dataset \
  --model_max_length 1500 \
  --cot_backup \
  --print
```

```bash
### For mutilple-choice questions, the dataset should be one of 
### ['aqua', 'sat', 'mmlu_mathematics'].
### We first try PoT and if the generated program is not executable, we shift to CoT
dataset='aqua'

python run_choice.py \
  --model "TIGER-Lab/MathOctopus-Coder-7B" \
  --shots 0 \
  --stem_flan_type "pot_prompt" \
  --match_answer "self"
  --stem_flan_type "" \
  --batch_size 8 \
  --dataset $dataset \
  --cot_backup \
  --print
```

### **Fine-tuning**

To train the 7B/13B model, run:

```bash
torchrun --nproc_per_node [$WORKER_GPU] \
 --master_addr [$WORKER_0_HOST] \
 --node_rank [$ROLE_INDEX] \
 --master_port [$WORKER_0_PORT] \
 --nnodes [$WORKER_NUM] \
train.py \
    --model_name_or_path "codellama/CodeLlama-7b-hf" \
    --data_path "TIGER-Lab/MathInstruct" \
    --bf16 True \
    --output_dir checkpoints/MathOctopus-Coder-7B \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000\
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True
```

To train the 34B/70B model, run:
```bash
torchrun --nproc_per_node [$WORKER_GPU] \
 --master_addr [$WORKER_0_HOST] \
 --node_rank [$ROLE_INDEX] \
 --master_port [$WORKER_0_PORT] \
 --nnodes [$WORKER_NUM] \
train.py \
    --model_name_or_path "codellama/CodeLlama-34b-hf" \
    --data_path "TIGER-Lab/MathInstruct" \
    --bf16 True \
    --output_dir checkpoints/MathOctopus-Coder-34B \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --deepspeed "ds_config/ds_config_zero3.json" \
    --tf32 True
```

## Prompt Format

If you want to do CoT:
```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
```

If you want to do PoT:
```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction} Let's write a program.

### Response:
```

## WebUI
We use [llama2-webui](https://github.com/liltom-eth/llama2-webui) as our ui bankend. To use webui for MathOctopus run:
```
pip install gradio
cd webui/llama2-webui
python3 MathOctopus.py --model_path your_model_path --backend_type transformers 
```



## **License**
Please check out the license of each subset in our curated dataset MathInstruct.
| Dataset Name 	| License Type   	|
|--------------	|----------------	|
| GSM8K        	| MIT            	|
| GSM8K-RFT    	| Non listed      |
| AQuA-RAT     	| Apache 2.0     	|
| MATH         	| MIT            	|
| TheoremQA    	| MIT            	|
| Camel-Math   	| Attribution-NonCommercial 4.0 International    	|
| NumGLUE      	| Apache-2.0          	|
| CrowdSourced (Lila)	| Attribution 4.0 International     	|
| MathQA       	| Apache-2.0     	|
| Our Curated   | MIT             |


## **Citation**

Please cite our paper if you use our data, model or code. Please also kindly cite the original dataset papers. 

```
@article{yue2023MathOctopus,
  title={MathOctopus: Building Math Generalist Models through Hybrid Instruction Tuning},
  author={Xiang Yue, Xingwei Qu, Ge Zhang, Yao Fu, Wenhao Huang, Huan Sun, Yu Su, Wenhu Chen},
  journal={arXiv preprint arXiv:2309.05653},
  year={2023}
}
```


