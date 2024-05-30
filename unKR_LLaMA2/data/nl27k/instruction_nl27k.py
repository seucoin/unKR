import json
import random
import csv
import numpy as np


def e2t(ent):
    text = ent.split(':')[-1]
    text = text.replace('_', ' ').title()
    return text

def e2d(ent):
    text = e2t(ent)
    enttype = ent.split(':')[1]
    a = text + ' is a ' + enttype
    return a

def r2t(rel):
    text = rel.split(':')[-1]
    return text




tail_lines_to_write_llama_lora, head_lines_to_write_llama_lora, conf_lines_to_write_llama_lora = [], [], []
def train_prompts(datapath):
    with open(datapath, "r", encoding = "utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split("\t")
            h, r, t, s = e2t(line[0]), r2t(line[1]), e2t(line[2]), line[3][:5]

            # tail entity prediction
            prompt = f"{h} {r}"
            response = t
            tmp_str = "{\n\"instruction\": \"" + prompt + "\",\n  \"input\": \"\",\n  \"output\": \""+ response + "\"\n}"
            tail_lines_to_write_llama_lora.append(tmp_str)

            # head entity prediction
            prompt = f"What {r} {t}?"
            response = h
            tmp_str = "{\n\"instruction\": \"" + prompt + "\",\n  \"input\": \"\",\n  \"output\": \""+ response + "\"\n}"
            head_lines_to_write_llama_lora.append(tmp_str)

            # confidence prediction
            prompt = f"What is the probability of the following fact being true: {h} {r} {t}?"
            response = s
            tmp_str = "{\n\"instruction\": \"" + prompt + "\",\n  \"input\": \"\",\n  \"output\": \""+ response + "\"\n}"
            conf_lines_to_write_llama_lora.append(tmp_str)


train_prompts('val.tsv')
train_prompts('train.tsv')

with open('nl27k_train_tail.json', "w") as f:
    tmp_str = "[\n" + ",\n".join(tail_lines_to_write_llama_lora) +"]"
    f.write(tmp_str)
 
with open('nl27k_train_conf.json', "w") as f:
    tmp_str = "[\n" + ",\n".join(conf_lines_to_write_llama_lora) +"]"
    f.write(tmp_str)


with open('nl27k_test_conf.csv', 'w', newline='') as fc, open('nl27k_test_tail.csv', 'w', newline='') as ft, open('test.tsv') as fr:
    writerc = csv.writer(fc, delimiter='\t')
    writert = csv.writer(ft, delimiter='\t')
    lines = csv.reader(fr, delimiter='\t')
    for line in lines:
        h, r, t, s = e2t(line[0]), r2t(line[1]), e2t(line[2]), line[3][:5]

        # tail entity prediction
        prompt = f"{h} {r}"
        response = t
        writert.writerow([prompt, response])

        # confidence prediction
        prompt = f"What is the probability of the following fact being true: {h} {r} {t}?"
        response = s
        writert.writerow([prompt, response])