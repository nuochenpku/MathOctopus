
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

#### **MGSM8KInstruct**

| Training Dataset      | En      | Sw      | Zh      | Bn      | De      | Es      | Fr      | Ja      | Ru      | Th      | Overall |
|:----------------------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|
| MGSM8KInstruct        | 7473    | 7472    | 7466    | 6539    | 7466    | 7470    | 7469    | 7471    | 7361    | 7473    | **73.6K**   |


#### **MSVAMP**

| Test Dataset      | En      | Sw      | Zh      | Bn      | De      | Es      | Fr      | Ja      | Ru      | Th      | Overall |
|:----------------------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|
| MSVAMP                | 1000    | 1000    | 1000    | 1000    | 1000    | 1000    | 1000    | 1000    | 1000    | 1000    | **10K**   |

#### Usage

Our dataset and models are all available at Huggingface.
🦑
🤗 [MathInstruct Dataset](https://huggingface.co/datasets/TIGER-Lab/MathInstruct)


Or you can directly download them from

##  Models
|  Base Model: LLama   	| Parallel-Training                                         	| Cross-Training                                                       	|
|-----	|---------------------------------------------------------------	|---------------------------------------------------------------------------	|
| 7B-LLaMA 2  	| 🐙 [MathOctopus-Parallel-7B](https://huggingface.co/Mathoctopus/Parallel_7B)   	| 🐙 [MathOctopus-Cross-7B](https://huggingface.co/Mathoctopus/Cross_7B)  	|
|| 🐙[MathOctopus-Parallel-xRFT-7B](https://huggingface.co/TIGER-Lab/MathOctopus-7B)|🐙[MathOctopus-Cross-xRFT-7B](https://huggingface.co/TIGER-Lab/MathOctopus-7B)|
| 13B-LLaMA 2 	| 🐙 [MathOctopus-Parallel-13B](https://huggingface.co/TIGER-Lab/MathOctopus-13B) 	| 🐙 [MathOctopus-Cross-13B](https://huggingface.co/TIGER-Lab/MathOctopus-Coder-13B) 	|
| 33B-LLaMA 1 	| 🐙 [MathOctopus-Parallel-33B](https://huggingface.co/TIGER-Lab/MathOctopus-13B)                                                             	| 🐙 [MathOctopus-Cross-33B](https://huggingface.co/TIGER-Lab/MathOctopus-Coder-34B) 	|
| 70B-LLaMA 2 	| Coming soon!	| Coming Soon!      |


### **Overall Results on MGSM8KInstruct**

| 7B Model                        | En      | Sw      | Zh      | Bn      | De      | Es      | Fr      | Ja      | Ru      | Th      | Overall |
|:--------------------------------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|
| MathOctupos<sup>C</sup>         | 52.0    | 23.6    | 31.6    | 18.8    | 38.0    | 39.2    | 36.4    | 27.2    | 33.6    | 21.6    | 32.2    |
| **xRFT**-MathOctupos<sup>C</sup>| 51.2    | 24.0    | 33.2    | 18.8    | 36.0    | 41.2    | 37.6    | 29.6    | 36.4    | 25.2    | 33.3    |
| MathOctupos<sup>P</sup>-LoRA    | 30.4    | 15.2    | 23.6    | 10.4    | 22.8    | 24.8    | 26.4    | 18.0    | 22.0    | 14.8    | 20.8    |
| MathOctupos<sup>P</sup>         | 52.4    | 39.2    | 38.4    | 28.8    | 44.8    | 42.4    | 43.6    | 36.0    | 39.6    | 34.4    | 40.0    |
| **xRFT**-MathOctupos<sup>P</sup>| 54.8    | 38.4    | 45.2    | 33.2    | 43.6    | 45.2    | 38.0    | 35.6    | 48.4    | 36.4    | 41.9    |
<p></p >

| 13B Model                       | En      | Sw      | Zh      | Bn      | De      | Es      | Fr      | Ja      | Ru      | Th      | Overall |
|:--------------------------------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|
| MathOctupos<sup>C</sup>         | 56.4    | 27.2    | 39.2    | 24.0    | 47.6    | 49.6    | 47.6    | 40.4    | 42.0    | 24.8    | 39.9    |
| **xRFT**-MathOctupos<sup>C</sup>| 53.6    | 28.0    | 45.2    | 21.2    | 48.0    | 46.4    | 46.0    | 35.2    | 45.6    | 28.8    | 39.8    |
| MathOctupos<sup>P</sup>         | 53.2    | 42.8    | 48.8    | 35.2    | 44.4    | 48.0    | 48.4    | 43.2    | 47.6    | 46.8    | 45.8    |
| **xRFT**-MathOctupos<sup>P</sup>| 51.6    | 46.0    | 51.2    | 42.0    | 49.2    | 53.2    | 49.6    | 39.6    | 47.6    | 46.0    | 47.6    |
<p></p >

| 30-34B Model                    | En      | Sw      | Zh      | Bn      | De      | Es      | Fr      | Ja      | Ru      | Th      | Overall |
|:--------------------------------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|
| MathOctupos<sup>C</sup>         | 55.6    | 24.4    | 36.0    | 19.2    | 40.4    | 51.2    | 44.4    | 27.2    | 37.2    | 21.6    | 35.7    |
| **xRFT**-MathOctupos<sup>C</sup>| 53.6    | 27.6    | 34.4    | 19.2    | 47.2    | 47.6    | 44.8    | 30.8    | 38.8    | 22.8    | 36.7    |
| MathOctupos<sup>P</sup>         | 56.4    | 46.8    | 52.0    | 35.2    | 47.2    | 53.2    | 48.0    | 39.2    | 45.6    | 41.2    | 46.5    |
| **xRFT**-MathOctupos<sup>P</sup>| 51.6    | 47.2    | 52.4    | 37.6    | 51.2    | 52.8    | 44.4    | 41.6    | 50.0    | 47.6    | 47.6    |


### **Overall Results on MSVAMP**

| 7B Model                        | En      | Sw      | Zh      | Bn      | De      | Es      | Fr      | Ja      | Ru      | Th      | Overall |
|:--------------------------------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|
| MathOctupos<sup>C</sup>         | 49.2    | 36.6    | 43.6    | 30.2    | 48.6    | 46.8    | 46.4    | 42.5    | 46.7    | 34.0    | 42.5    |
| **xRFT**-MathOctupos<sup>C</sup>| 49.9    | 37.7    | 43.3    | 32.9    | 46.5    | 47.6    | 47.3    | 42.7    | 46.6    | 36.2    | 43.1    |
| MathOctupos<sup>P</sup>-LoRA    | 30.4    | 15.2    | 23.6    | 10.4    | 22.8    | 24.8    | 26.4    | 18.0    | 22.0    | 14.8    | 20.8    |
| MathOctupos<sup>P</sup>         | 46.5    | 40.1    | 42.5    | 29.1    | 43.5    | 45.4    | 46.0    | 42.5    | 45.4    | 35.7    | 41.7    |
| **xRFT**-MathOctupos<sup>P</sup>| 46.8    | 42.3    | 43.2    | 32.8    | 43.1    | 44.5    | 45.3    | 43.2    | 42.1    | 40.5    | 42.4    |
<p></p >

| 13B Model                       | En      | Sw      | Zh      | Bn      | De      | Es      | Fr      | Ja      | Ru      | Th      | Overall |
|:--------------------------------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|
| MathOctupos<sup>C</sup>         | 56.6    | 40.4    | 49.0    | 30.3    | 50.9    | 54.2    | 54.7    | 46.3    | 52.4    | 35.7    | 47.1    |
| **xRFT**-MathOctupos<sup>C</sup>| 52.9    | 41.9    | 49.2    | 34.1    | 50.5    | 52.8    | 51.5    | 45.8    | 50.2    | 35.7    | 46.5    |
| MathOctupos<sup>P</sup>         | 50.7    | 43.4    | 42.6    | 31.8    | 48.4    | 49.4    | 50.6    | 41.1    | 46.9    | 39.3    | 44.4    |
| **xRFT**-MathOctupos<sup>P</sup>| 44.6    | 43.4    | 46.4    | 34.2    | 47.7    | 48.2    | 49.9    | 43.1    | 48.2    | 39.5    | 44.5    |
<p></p >

| 30-34B Model                    | En      | Sw      | Zh      | Bn      | De      | Es      | Fr      | Ja      | Ru      | Th      | Overall |
|:--------------------------------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|
| MathOctupos<sup>C</sup>         | 51.5    | 42.1    | 46.2    | 23.2    | 50.5    | 52.1    | 52.9    | 42.2    | 50.5    | 33.4    | 44.5    |
| **xRFT**-MathOctupos<sup>C</sup>| 48.1    | 42.8    | 43.6    | 23.3    | 48.7    | 50.0    | 48.9    | 43.4    | 44.6    | 35.5    | 42.9    |
| MathOctupos<sup>P</sup>         | 56.4    | 46.8    | 52.0    | 35.2    | 47.2    | 53.2    | 48.0    | 39.2    | 45.6    | 41.2    | 46.5    |
| **xRFT**-MathOctupos<sup>P</sup>| 48.0    | 42.3    | 46.1    | 36.2    | 47.5    | 48.5    | 48.3    | 45.8    | 47.2    | 41.2    | 45.1    |


### **MathOctupos in English**

| Models                          | GSM8K   | SVAMP   |
|:--------------------------------|:--------|:--------|
| LLaMA 2-7B                      | 42.4    | 38.3    |
| MathOctupos<sup>P</sup>-7B      | 49.3    | 46.8    |
| MathOctupos<sup>C</sup>-7B      | 50.8    | 49.3    |
| LLaMA 2-13B                     | 51.0    | 50.9    |
| MathOctupos<sup>P</sup>-13B     | 55.5    | 52.1    |
| MathOctupos<sup>C</sup>-13B     | 56.6    | 56.6    |
| LLaMA 1-33B                     | 50.0    | 49.0    |
| MathOctupos<sup>P</sup>-33B     | 56.0    | 52.5    |
| MathOctupos<sup>C</sup>-33B     | 53.7    | 51.5    |


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


