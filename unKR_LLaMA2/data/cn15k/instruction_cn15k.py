import json
import random
import csv
import numpy as np


id2ent, id2rel, rel2text = dict(), dict(), dict()
with open('entity_id.csv', 'r') as f:
    lines = csv.reader(f)
    for line in lines:
        ent, id = line[0], line[1]
        id2ent[id] = ent
        # print(ent)
with open('relation_id.csv', 'r') as f:
    lines = csv.reader(f)
    for line in lines:
        rel, id = line[0], line[1]
        id2rel[id] = rel
with open('relation_text.csv') as f:
    lines = csv.reader(f)
    for line in lines:
        rel, txt = line[0], line[1]
        # print(rel, txt)
        rel2text[rel] = txt




# tail + head + conf
tail_lines_to_write_llama_lora, head_lines_to_write_llama_lora, conf_lines_to_write_llama_lora = [], [], []
def train_prompts(datapath):
    with open(datapath, "r", encoding = "utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split("\t")
            h, r, t, s = id2ent[line[0]], rel2text[id2rel[line[1]]], id2ent[line[2]], line[3][:5]

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


with open('cn15k_train_tail.json', "w") as f:
    tmp_str = "[\n" + ",\n".join(tail_lines_to_write_llama_lora) +"]"
    f.write(tmp_str)
 
with open('cn15k_train_conf.json', "w") as f:
    tmp_str = "[\n" + ",\n".join(conf_lines_to_write_llama_lora) +"]"
    f.write(tmp_str)


with open('cn15k_test_conf.csv', 'w', newline='') as fc, open('cn15k_test_tail.csv', 'w', newline='') as ft, open('test.tsv') as fr:
    writerc = csv.writer(fc, delimiter='\t')
    writert = csv.writer(ft, delimiter='\t')
    lines = csv.reader(fr, delimiter='\t')
    for line in lines:
        h, r, t, s = id2ent[line[0]], rel2text[id2rel[line[1]]], id2ent[line[2]], line[3][:5]

        # tail entity prediction
        prompt = f"{h} {r}"
        response = t
        writert.writerow([prompt, response])

        # confidence prediction
        prompt = f"What is the probability of the following fact being true: {h} {r} {t}?"
        response = s
        writert.writerow([prompt, response])