# unKR_LLaMA
We perform instruction tuning with [LLaMA-2-7B](https://huggingface.co/meta-llama/Llama-2-7b) by LoRA. We design instruction prompts for confidence prediction and tail entity prediction tasks. Through instruction tuning, the large language model can obtaine the reasoning ability on UKGs.


## How to run

### 1. Prepare the datasets
We use CN15K and NL27k for LLM instruction tuning. You can use the preprocessed data we provide in folder ./data, or you can refer to the ./data/nl27k/instruction_nl27k.py file to build your own datasets.


### 2. LLaMA Instruction Tuning
2.1 Firstly, put [LLaMA2-7B](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) model files under ./models/LLaMA-HF/.

2.2 In our experiments, we utilized A100 GPU for training. Set the `DATA_PATH` as your own dataset path and `OUTPUT_DIR` for model output path. And then run 

```
python lora_finetune.py
```


### 3. LLaMA Infernce
Set the `LORA_WEIGHTS` as the path of saved checkpoints and `TEST_DATA_PATH` as the path of test dataset. And run 
```
python lora_infer.py
```
We provide our trained models at [here](https://drive.google.com/drive/folders/1_vitwjZt0A5QZRwQr02X3bNm6HO6Snp5?usp=drive_link).
