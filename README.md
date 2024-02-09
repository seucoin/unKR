# unKR: A Python Library for Uncertain Knowledge Graph Reasoning by Representation Learning
<p align="center">
    <a href="https://pypi.org/project/unKR/">
        <img alt="Pypi" src="https://img.shields.io/pypi/v/unKR">
    </a>
    <a href="https://github.com/seucoin/unKR/blob/main/LICENSE">
        <img alt="Pypi" src="https://img.shields.io/badge/license-Apache--2.0-yellowgreen">
    </a>
    <!-- <a href="">
        <img alt="LICENSE" src="https://img.shields.io/badge/license-MIT-brightgreen">
    </a> -->
    <a href="https://seucoin.github.io/unKR/">
        <img alt="Documentation" src="https://img.shields.io/badge/Doc-online-blue">
    </a>
</p>
<p align="center">
    <b> English | <a href="https://github.com/seucoin/unKR/blob/main/README_CN.md">中文</a></b>
</p>

unKR is an python library for **un**certain **K**nowledge graph (UKG) **R**easoning based on the [PyTorch Lightning](https://www.pytorchlightning.ai/). It provides a unifying workflow to implement a variety of uncertain knowledge graph representation learning models to complete UKG reasoning. unKR consists of five modules: 1) Data Processor handles low-level dataset parsing and negative sampling, then generates mini-batches of data; 2) Model Hub implements the model algorithms, containing the scoring function and loss function; 3) Trainer conducts iterative training and validation; 4) **Evaluator** provides confidence prediction and link prediction tasks to evaluate models' performance; 5) Controller controls the training worklow, allowing for early stopping and model saving. These modules are decoupled and independent, making unKR highly modularized and extensible. Detailed documentation of the unKR is available at [here](https://seucoin.github.io/unKR/).

unKR core development team will provide long-term technical support, and developers are welcome to discuss the work and initiate questions using `issue`.



<h3 align="center">
    <img src="pics/unKR.svg", width="1000">
</h3>
<!-- <p align="center">
    <a href=""> <img src="pics/unKR.svg" width="1000"/></a>
<p> -->



## 💻 Demo
This is a demo shows the training and testing process of [PASSLEAF](https://ojs.aaai.org/index.php/AAAI/article/view/16522) model with unKR.
<!-- ![demo](./pics/demo.gif) -->

<img src="pics/demo.gif">

<!-- <img src="pics/demo.gif" width="900" height="500" align=center> -->


## Datasets
unKR provides three public UKG datasets including CN15K, NL27K, and PPI5K. The following table shows the source, the number of entities, relations, and facts of each dataset.

| Dataset |   Source   | #Entity  | #Relation | #Fact |
|:-------:|:----------:|:---------:|:---------:|:-----------:|
|  CN15K  | ConceptNet |   15000   |    36     |   241158    |
|  NL27K  |    NELL    |   27221   |    404    |   175412    |
|  PPI5K  |   STRING   |   4999    |     7     |   271666    |


## 📝 Models
Now, nine uncertain knowledge graph representation learning models are available and they can be divided to two types: normal and few-shot models.


|   Type   |                                                                                                                                                                                                                                                    Model                                                                                                                                                                                                                                                     |
|:--------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|  Normal  | [BEURrE](https://aclanthology.org/2021.naacl-main.68), [FocusE](https://www.ijcai.org/proceedings/2021/395), [GTransE](https://link.springer.com/chapter/10.1007/978-3-030-39878-1_16), [PASSLEAF](https://ojs.aaai.org/index.php/AAAI/article/view/16522), [UKGE](https://ojs.aaai.org/index.php/AAAI/article/view/4210), [UKGsE](https://www.sciencedirect.com/science/article/abs/pii/S0020025522007885), [UPGAT](https://link.springer.com/chapter/10.1007/978-3-031-33377-4_5) |
| Few-shot |                                                                                                                                                                                [GMUC](https://link.springer.com/chapter/10.1007/978-3-030-73194-6_18), [GMUC+](https://link.springer.com/chapter/10.1007/978-981-19-7596-7_2)                                                                                                                                                                                |


## Reproduce Results
unKR determines two tasks, confidence prediction and link prediction, to evaluate models' ability of UKG reasoning. For confidence prediction task, MSE (Mean Squared Error) and MAE (Mean Absolute Error) are reported. For link prediction task, Hits@k(k=1,3,10), MRR (Mean Reciprocal Rank), MR (Mean Rank) under both raw and filterd settings are reported.

Here are the reproduce results of nine models on NL27K dataset with unKR. See more results at [here](https://seucoin.github.io/unKR/result.html).

### Confidence prediction
<table>
    <thead>
        <tr>
            <th>Type</th>
            <th>Model</th>
            <th>MSE</th>
            <th>MAE </th>
        </tr>
    </thead>
    <tbody align="center" valign="center">
        <tr>
            <td rowspan="10">Normal</td>
            <td>BEUrRE</td>
            <td>0.08920 </td>
            <td>0.22194  </td>
        </tr>
        <tr>
            <td>PASSLEAF_ComplEx</td>
            <td>0.02434 </td>
            <td>0.05176  </td>
        </tr>
        <tr>
            <td>PASSLEAF_DistMult</td>
            <td>0.02309 </td>
            <td>0.05107  </td>
        </tr>
        <tr>
            <td>PASSLEAF_RotatE</td>
            <td>0.01949 </td>
            <td>0.06253  </td>
        </tr>
        <tr>
            <td>UKGElogi</td>
            <td>0.02861 </td>
            <td>0.05967  </td>
        </tr>
        <tr>
            <td>UKGElogiPSL</td>
            <td>0.02868 </td>
            <td>0.05966  </td>
        </tr>
        <tr>
            <td>UKGErect</td>
            <td>0.03344 </td>
            <td>0.07052  </td>
        </tr>
        <tr>
            <td>UKGErectPSL</td>
            <td>0.03326 </td>
            <td>0.07015 </td>
        </tr>
        <tr>
            <td>UKGsE</td>
            <td>0.12202 </td>
            <td>0.27065  </td>
        </tr>
        <tr>
            <td>UPGAT</td>
            <td>0.02922 </td>
            <td>0.10107  </td>
        </tr>
        <tr>
            <td rowspan="2">Few-shot</td>
            <td>GMUC</td>
            <td>0.01300 </td>
            <td>0.08200  </td>
        </tr>
        <tr>
            <td>GMUC+</td>
            <td>0.01300 </td>
            <td>0.08600  </td>
        </tr>
    </tbody>
</table>

### Link prediction (filter on high-confidence test data)
<table>
    <thead>
        <tr>
            <th>Type</th>
            <th>Model</th>
            <th>Hits@1</th>
            <th>Hits@3</th>
            <th>Hits@10</th>
            <th>MRR</th>
            <th>MR</th>
        </tr>
    </thead>
    <tbody align="center" valign="center">
        <tr>
            <td rowspan="12">Normal</td>
            <td>BEUrRE</td>
            <td>0.156 </td>
            <td>0.385 </td>
            <td>0.543 </td>
            <td>0.299 </td>
            <td>488.051 </td>
        </tr>
        <tr>
            <td>FocusE</td>
            <td>0.814 </td>
            <td>0.918 </td>
            <td>0.957 </td>
            <td>0.870 </td>
            <td>384.471 </td>
        </tr>
        <tr>
            <td>GTransE</td>
            <td>0.222 </td>
            <td>0.366 </td>
            <td>0.493 </td>
            <td>0.316 </td>
            <td>1377.564 </td>
        </tr>
        <tr>
            <td>PASSLEAF_ComplEx</td>
            <td>0.669 </td>
            <td>0.786 </td>
            <td>0.876 </td>
            <td>0.741 </td>
            <td>138.808 </td>
        </tr>
        <tr>
            <td>PASSLEAF_DistMult</td>
            <td>0.627 </td>
            <td>0.754 </td>
            <td>0.856 </td>
            <td>0.707 </td>
            <td>138.781 </td>
        </tr>
        <tr>
            <td>PASSLEAF_RotatE</td>
            <td>0.687 </td>
            <td>0.816 </td>
            <td>0.884 </td>
            <td>0.762 </td>
            <td>50.776 </td>
        </tr>
        <tr>
            <td>UKGElogi</td>
            <td>0.526 </td>
            <td>0.670 </td>
            <td>0.805 </td>
            <td>0.622 </td>
            <td>153.632 </td>
        </tr>
        <tr>
            <td>UKGElogiPSL</td>
            <td>0.525 </td>
            <td>0.673 </td>
            <td>0.812 </td>
            <td>0.623 </td>
            <td>168.029 </td>
        </tr>
        <tr>
            <td>UKGErect</td>
            <td>0.509 </td>
            <td>0.662 </td>
            <td>0.807 </td>
            <td>0.609 </td>
            <td>126.011 </td>
        </tr>
        <tr>
            <td>UKGErectPSL</td>
            <td>0.500 </td>
            <td>0.647 </td>
            <td>0.800 </td>
            <td>0.599 </td>
            <td>125.233 </td>
        </tr>
        <tr>
            <td>UKGsE</td>
            <td>0.038 </td>
            <td>0.073 </td>
            <td>0.130 </td>
            <td>0.069 </td>
            <td>2329.501 </td>
        </tr>
        <tr>
            <td>UPGAT</td>
            <td>0.618 </td>
            <td>0.751 </td>
            <td>0.862 </td>
            <td>0.701 </td>
            <td>69.120 </td>
        </tr>
        <tr>
            <td rowspan="2">Few-shot</td>
            <td>GMUC</td>
            <td>0.335 </td>
            <td>0.465 </td>
            <td>0.592 </td>
            <td>0.425 </td>
            <td>58.312 </td>
        </tr>
        <tr>
            <td>GMUC+</td>
            <td>0.338 </td>
            <td>0.486 </td>
            <td>0.636 </td>
            <td>0.438 </td>
            <td>45.774 </td>
        </tr>
    </tbody>
</table>

<br>

# 🛠️ Usage

## Installation

**Step1** Create a virtual environment using ```Anaconda``` and enter it.

```bash
conda create -n unKR python=3.8
conda activate unKR
pip install -r requirements.txt
```

**Step2**  Install package.
+ Install from source
```bash
git clone https://github.com/seucoin/unKR.git
cd unKR
python setup.py install
```
+ Install by pypi
```bash
pip install unKR
```

## Data Format
For normal models, `train.tsv`, `val.tsv`, and `test.tsv` are required. 



For few-shot models, `train_tasks.json`, `dev_tasks.json`, `test_tasks.json` and `path_graph` are required.

```
UKGE model:
    softloic.tsv: (ent1, rel, ent2, score)
GMUC, GMUC+ models:
    train/dev/test_tasks.json: {rel:[[ent1, rel, ent2, score], ...]}
    path_graph: (ent1, rel, ent2, score)
    ontology.csv: (number, h, rel, t)
```

`train.tsv`: All triples used for training and corresponding confidence scores in the format`(ent1, rel, ent2, score)`, one quaternion per line.

`val.tsv`: All triples used for validation and corresponding confidence scores in the format`(ent1, rel, ent2, score)`, one quaternion per line.

`test.tsv`: All triples used for testing and corresponding confidence scores in the format`(ent1, rel, ent2, score)`, one quaternion per line.

In [UKGE](https://ojs.aaai.org/index.php/AAAI/article/view/4210), the`softlogic.tsv`file is also required.

`softlogic.tsv`: All triples inferred by PSL rule and corresponding inferred confidence scores in the format`(ent1, rel, ent2, score)`, one quaternion per line.

In [GMUC](https://link.springer.com/chapter/10.1007/978-3-030-73194-6_18), [GMUC+](https://link.springer.com/chapter/10.1007/978-981-19-7596-7_2), the following five data files are also needed.

`train/dev/test_tasks.json`: Few-shot dataset with one task per relation in the format`{rel:[[ent1, rel, ent2, score], ...]}`. The key of the dictionary is the task name and the values are all the quaternions under the task.

`path_graph`: All data except training, validation and testing tasks, i.e. background knowledge, in the format`(ent1, rel, ent2, score)`. Each line represents a quaternion.

`ontology.csv`: Ontology knowledge data required for the GMUC+ model, in the format`(number, h, rel, t)`, one ontology knowledge per line. There are four types of **rel**, which includes **is_A**, **domain**, **range**, and **type**.

- c1 **is_A** c2: c1 is a **subclass** of c2;
- c1 **domain** c2: the **definition domain** of c1 is c2;
- c1 **range** c2: the **value domain** of c1 is c2;
- c1 **type** c2: the **type** of c1 is c2.



## Parameter Adjustment
In the [config](https://github.com/seucoin/unKR/tree/main/config) file, we provide parameter profiles of the reproduced results, and the following parameters can be adjusted for specific use.

```
parameters:
  confidence_filter:  #whether to perform high-confidence filtering
    values: [0, 0.7]
  emb_dim:
    values: [128, 256, 512...]
  lr:
    values: [1.0e-03, 3.0e-04, 5.0e-06...]
  num_neg:
    values: [1, 10, 20...]
  train_bs:
    values: [64, 128, 256...]
```


## Model Training
```bash
python main.py --load_config --config_path <your-config>
```

## Model Testing
```bash
python main.py --test_only --checkpoint_dir <your-model-path>
```

## Model Customization
If you want to personalise your own model using unKR, you need to define the following Functions/Classes.

`data`: Implement data processing functions, including `DataPreprocess`, `Sampler` and `KGDataModule`.
```
DataPreprocess.py: 
    class unKR.data.DataPreprocess.<your-model-name>BaseSampler
    class unKR.data.DataPreprocess.<your-model-name>Data
Sampler:
    class unKR.data.Sampler.<your-model-name>Sampler
    class unKR.data.Sampler.<your-model-name>TestSampler
KGDataModule.py: 
    class unKR.data.KGDataModule.<your-model-name>DataModule
```

`lit_model`: Implement model training, validation, and testing functions.
```
<your-model-name>LitModel.py:
    class unKR.lit_model.<your-model-name>LitModel.<your-model-name>LitModel
```
`loss`: Implement loss functions.
```
<your-model-name>_Loss.py:
    class unKR.loss.<your-model-name>_Loss.<your-model-name>_Loss
```
`model`: Implement model framework functions, classified as `UKGModel` and `FSUKGModel` based on whether it is a few-shot model.
```
<your-model-name>.py:
    class unKR.model.UKGModel/FSUKGModel.<your-model-name>.<your-model-name>
```
`config`: Implement parameter settings.
```
<your-model-name>_<dataset-name>.yaml:
    data_class, litmodel_name, loss_name, model_name, test_sampler_class, train_sampler_class
```
`demo`: Implement the model run file.
```
<your-model-name>demo.py
```

<br>



# 😊 unKR Core Team

**Southeast University**: Jingting Wang, Tianxing Wu, Shilin Chen, Yunchang Liu, Shutong Zhu, Wei Li, Jingyi Xu, Guilin Qi.


