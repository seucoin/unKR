import os
#os.system("pip install datasets")
#os.system("pip install deepspeed")
#os.system("pip install accelerate")
#os.system("pip install transformers>=4.28.0")
import sys
import torch
import argparse
import pandas as pd
from peft import PeftModel
import transformers
# import gradio as gr
# assert (
#     "LlamaTokenizer" in transformers._import_structure["models.llama"]
# ), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("Llama-2-7b-chat-hf")
LOAD_8BIT = False
BASE_MODEL = "./model/Llama-2-7b-chat-hf"
LORA_WEIGHTS = "./model/cn15k_conf/"
TEST_DATA_PATH = "./data/cn15k_test.csv"

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass
if device == "cuda":
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        # load_in_8bit=LOAD_8BIT,
        # torch_dtype=torch.float16,
        # device_map="auto",
    ).cuda()
    model = PeftModel.from_pretrained(
        model,
        LORA_WEIGHTS,
        # torch_dtype=torch.float16,
    ).cuda()
elif device == "mps":
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(
        model,
        LORA_WEIGHTS,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, device_map={"": device}, low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(
        model,
        LORA_WEIGHTS,
        device_map={"": device},
    )
def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Input:
{input}
### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Response:"""
# if not LOAD_8BIT:
    # model.half()  # seems to fix bugs for some users.
model.eval()
if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)
def evaluate(
    instruction,
    input=None,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=256,
    **kwargs,
):
    prompt = generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        do_sample = True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return output.split("### Response:")[1].strip()

if __name__ == "__main__":
    num = 0
    result = 0
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        content = f.readlines()
        # print(content)
    with open("./result/cn15k_checkpoint-10900.txt", 'w') as f2:
        for line in content:
            query = line.split('\t')
            print('query: ', query[0])
            print('truth: ', float(query[1].strip().strip(".")[:5]))
            print("-------------------------------")
            real = float(query[1].strip().strip(".")[:5])
            try:
                pre = evaluate(query[0])
                # pre = float(pre[:pre.index("</s>")])
                print('predicted: ')
                print(pre)
                print("###############################")
                # result=abs(pre-real)*abs(pre-real)+result
                # num = num+1
                f2.write(query[0]+'\t'+str(real)+'\t'+str(pre)+'\n')
            except:
                print("###############################")
                f2.write(query[0]+'\t'+str(real)+'\t'+''+'\n')


    
