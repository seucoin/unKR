# unKR: An Open Source Toolkit for Uncertain Knowledge Graph Representation Learning
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

unKR is an open source toolkit for Uncertain Knowledge Graph Representation Learning(UKRL). 
It is based on the [PyTorch Lightning](https://www.pytorchlightning.ai/) framework to decouple the workflow of the UKRL models in order to implement multiple Uncertain Knowledge Graph Embedding(UKGE) methods, which in turn assist knowledge graph complementation, inference and other tasks.
The tool provides code implementations and results of various existing UKGE models, and provides users with detailed [technical documentation](https://seucoin.github.io/unKR/index.html).
<br>



# 🔖 Overview

<h3 align="center">
    <img src="pics/overview.png", width="600">
</h3>
<!-- <p align="center">
    <a href=""> <img src="pics/overview.png" width="400"/></a>
<p> -->
（图片待修改）

unKR toolkit is an efficient implementation for Uncertain Knowledge Graph Representation Learning(URKL) based on the [PyTorch Lightning](https://www.pytorchlightning.ai/) framework. 
It provides a refinement module process that can implement a variety of Uncertain Knowledge Graph Embedding(UKGE) models, including UKG data preprocessing(Sampler for negative sampling), model implementation base module, and model training, validation, and testing modules. 
These modules are widely used in different UKGE models, facilitating users to quickly construct their own models.

There are nine different models available, divided according to whether they are small-sample models or not. 
unKR has validated the tool on three datasets with seven different evaluation metrics, and the details of the models will be discussed in the following sections.

unKR core development team will provide long-term technical support for the toolkit, and developers are welcome to discuss the work and initiate questions using `issue`.

Detailed documentation of the unKR technology and results is available at [📋](https://seucoin.github.io/unKR/).

<br>


# 📝 Models
unKR implements nine UKGE methods that partition the model based on whether it is a small-sample model or not. The available models are as below.

|          Category           |                                                                                                                                                                                                                                                     Model                                                                                                                                                                                                                                                     |
|:---------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|   Non-small-sample model    | [BEURrE](https://aclanthology.org/2021.naacl-main.68), [FocusE](https://www.ijcai.org/proceedings/2021/395), [GTransE](https://link.springer.com/chapter/10.1007/978-3-030-39878-1_16), [PASSLEAF]( http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-rela), [UKGE](https://ojs.aaai.org/index.php/AAAI/article/view/4210), [UKGsE](https://www.sciencedirect.com/science/article/abs/pii/S0020025522007885), [UPGAT](https://link.springer.com/chapter/10.1007/978-3-031-33377-4_5) |
|     Small-sample model      |                                                                                                                                                                                [GMUC](https://link.springer.com/chapter/10.1007/978-3-030-73194-6_18), [GMUC+](https://link.springer.com/chapter/10.1007/978-981-19-7596-7_2)                                                                                                                                                                                 |




## Datasets
unKR provides three different sources of UKG datasets including CN15K, NL27K, and PPI5K. The following table respectively shows the source of the datasets and the number of entities, relationships, and quaternions they contain.

| Dataset |   Source   | Entities  | Relations | Quaternions |
|:-------:|:----------:|:---------:|:---------:|:-----------:|
|  CN15K  | ConceptNet |   15000   |    36     |   241158    |
|  NL27K  |    NELL    |   27221   |    404    |   175412    |
|  PPI5K  |   STRING   |   4999    |     7     |   271666    |

Organize the three datasets, here are the three data files common to all models.

`train.tsv`: All triples used for training and corresponding confidence scores in the format`(ent1, rel, ent2, score)`, one quaternion per line.

`val.tsv`: All triples used for validation and corresponding confidence scores in the format`(ent1, rel, ent2, score)`, one quaternion per line.

`test.tsv`: All triples used for testing and corresponding confidence scores in the format`(ent1, rel, ent2, score)`, one quaternion per line.

In [UKGE](https://ojs.aaai.org/index.php/AAAI/article/view/4210), the`softlogic.tsv`file is also required.

`softlogic.tsv`: All triples inferred by PSL rule and corresponding inferred confidence scores in the format`(ent1, rel, ent2, score)`, one quaternion per line.

In [GMUC](https://link.springer.com/chapter/10.1007/978-3-030-73194-6_18), [GMUC+](https://link.springer.com/chapter/10.1007/978-981-19-7596-7_2), the following five data files are also needed.

`train/dev/test_tasks.json`: Small sample dataset with one task per relation in the format`{rel:[[ent1, rel, ent2, score], ...]}`. The key of the dictionary is the task name and the values are all the quaternions under the task.

`path_graph`: All data except training, validation and testing tasks, i.e. background knowledge, in the format`(ent1, rel, ent2, score)`. Each line represents a quaternion.

`ontology.csv`: Ontology knowledge data required for the GMUC+ model, in the format`(number, h, rel, t)`, one ontology knowledge per line. There are four types of **rel**, which includes **is_A**, **domain**, **range**, and **type**.

- c1 **is_A** c2: c1 is a **subclass** of c2;
- c1 **domain** c2: the **definition domain** of c1 is c2;
- c1 **range** c2: the **value domain** of c1 is c2;
- c1 **type** c2: the **type** of c1 is c2.


## Reproduced Results
unKR uses confidence prediction and link prediction tasks for model evaluation in seven different metrics, MSE, MAE, Hits@k(k=1,3,10), MRR, MR, WMRR, and WMR, with raw and filter settings. 
In addition, unKR adopts a high-confidence filter(set the filter value to 0.7) method for the evaluation.

Here are the reproduced model results on NL27K dataset using unKR as below. See more results in [here](https://seucoin.github.io/unKR/result.html).

### Raw
|   Model   | Confidence Filter(0.7) | MSE         | MAE         | Hits@1      | Hits@3      | Hits@10     | MRR         | MR          | WMRR        | WMR          |
|:---------:|:----------------------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:------------:|
|  BEUrRE   | yes                    | 0.089199997 | 0.221935004 | 0.114580996 | 0.294553995 | 0.436170012 | 0.231920004 | 549.5641479 | 0.237371996 | 532.7346802  |
|  BEUrRE   | no                     | 0.089199997 | 0.221935004 | 0.086361997 | 0.236710995 | 0.357560009 | 0.186324 | 937.3199463  | 0.211959004 | 745.0280762  |
|  FocusE   | yes                    | 1704.796143 | 32.64796448 | 0.393422008 | 0.53679502 | 0.659085989 | 0.48723501    | 436.3726196 | 0.490559012 | 430.8166809  |
|  FocusE   | no                     | 1704.796143 | 32.64796448 | 0.372880012 | 0.517742991 | 0.646003008 | 0.468558997 | 515.3634033 | 0.480058998  | 473.3652649  |
|   GMUC    | yes                    | 0.01200     | 0.08200     | 0.28100     | 0.40000     | 0.54000     | 0.36800     | 62.00500    | 0.36800     | 61.84900     |
|   GMUC    | no                     | 0.01300     | 0.08200     | 0.28700     | 0.40900     | 0.53600     | 0.37500     | 71.48400    | 0.37500     | 71.44700     |
|   GMUC+   | yes                    | 0.01500     | 0.10200     | 0.29000     | 0.42000     | 0.57300     | 0.43800     | 45.77400    | 0.38400     | 49.80800     |
|   GMUC+   | no                     | 0.01300     | 0.08600     | 0.29900     | 0.44800     | 0.58200     | 0.40100     | 49.41800    | 0.40100     | 49.10700     |
|  GTransE  | yes                    | 39.83544    | 5.12528     | 0.16800     | 0.28700     | 0.40700     | 0.25000     | 1434.63400  | 0.25300     | 1435.39700   |
|  GTransE  | no                     | 39.83544    | 5.12528     | 0.13674     | 0.24476     | 0.35250     | 0.21145     | 2014.54199  | 0.23173     | 1749.63757   |
| PASSLEAF  | yes                    | 0.023089999 | 0.051066 | 0.399278015     | 0.53747803     | 0.658012986     | 0.490554988     | 184.2464447   | 0.496309012     | 182.1270447    |
| PASSLEAF  | no                     | 0.023089999 | 0.051066 | 0.368676007     | 0.501568019     | 0.621562004     | 0.45787999     | 214.0929108   | 0.477825999     | 198.7584991    |
| UKGE(PSL) | yes                    | 0.028677    | 0.059657998 | 0.384539992     | 0.520008028     | 0.644250989     | 0.47556299     | 212.7815704   | 0.48159799     | 210.7985077    |
| UKGE(PSL) | no                     | 0.028677    | 0.059657998 | 0.346516013     | 0.477055997     | 0.598973989     | 0.435442001     | 265.9153442   | 0.459026992     | 240.2348633    |
|   UKGsE   | yes                    | 0.12202     | 0.27065     | 0.03543     | 0.06695     | 0.12376     | 0.06560     | 2378.45581  | 0.06561     | 2336.46582   |
|   UKGsE   | no                     | 0.12202     | 0.27065     | 0.03000     | 0.05800     | 0.10800     | 0.05700     | 3022.76900  | 0.06100     | 2690.49600   |
|   UPGAT   | yes                    | 0.02922     | 0.10107     | 0.37900     | 0.52000     | 0.64500     | 0.47300     | 114.65800   | 0.47700     | 113.82700    |
|   UPGAT   | no                     | 0.02922     | 0.10107     | 0.33900     | 0.46700     | 0.58600     | 0.42600     | 166.16900   | 0.45200     | 141.35800    |

### Filter

|       Model        | Confidence Filter(0.7) | MSE         | MAE         | Hits@1      | Hits@3      | Hits@10     | MRR         | MR          | WMRR        | WMR          |
|:------------------:|:----------------------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:------------:|
|       BEUrRE       | yes                    | 0.089199997 | 0.221935004 | 0.156060994 | 0.385224015 | 0.543040991 | 0.299324006 | 488.0508423 | 0.306418985 | 471.7837219  |
|       BEUrRE       | no                     | 0.089199997 | 0.221935004 | 0.11693 | 0.306755006 | 0.441855013 | 0.237922996 | 874.3798218  | 0.272213995 | 683.2228394  |
|       FocusE       | yes                    | 1704.796143 | 32.64796448 | 0.813781023 | 0.91791898 | 0.956959009 | 0.869915009 | 384.4714966  | 0.870938003 | 379.7612305  |
|       FocusE       | no                     | 1704.796143 | 32.64796448 | 0.760794997| 0.88371098 | 0.945631981  | 0.829499006 | 459.7191162 | 0.849004984 | 420.1532288  |
|        GMUC        | yes                    | 0.01200     | 0.08200     | 0.33500     | 0.46500     | 0.59200     | 0.42500     | 58.31200    | 0.42600     | 58.09700     |
|        GMUC        | no                     | 0.01300     | 0.08200     | 0.34400     | 0.46200     | 0.59200     | 0.43000     | 67.92000    | 0.43200     | 67.81300     |
|       GMUC+        | yes                    | 0.01500     | 0.10200     | 0.33800     | 0.48600     | 0.63600     | 0.43800     | 45.77400    | 0.43800     | 45.68200     |
|       GMUC+        | no                     | 0.01300     | 0.08600     | 0.37100     | 0.50500     | 0.63800     | 0.46300     | 45.87400    | 0.46500     | 45.49500     |
|      GTransE       | yes                    | 39.83544    | 5.12528     | 0.22200     | 0.36600     | 0.49300     | 0.31600     | 1377.56400  | 0.31900     | 1378.50500   |
|      GTransE       | no                     | 39.83544    | 5.12528     | 0.17914     | 0.30818     | 0.42461     | 0.26475     | 1957.77161  | 0.29136     | 1692.88000   |
| PASSLEAF(DistMult) | yes                    | 0.023089999 | 0.051066 | 0.627074003     | 0.754245996     | 0.855943978     | 0.706506014     | 138.7805939   | 0.717220008     | 137.8641663    |
| PASSLEAF(DistMult) | no                     | 0.023089999 | 0.051066 | 0.552658021     | 0.676356971     | 0.782598972     | 0.633262992     | 163.501358   | 0.67601198     | 151.5425568    |
|     UKGE(PSL)      | yes                    | 0.028677    | 0.059657998 | 0.525472999     | 0.672945976     | 0.812121987     | 0.622503996     | 168.028595   | 0.631505013     | 167.3438416    |
|     UKGE(PSL)      | no                     | 0.028677    | 0.059657998 | 0.461093992     | 0.600184977     | 0.733717978     | 0.555122972     | 216.390625   | 0.59375298     | 194.0056    |
|       UKGsE        | yes                    | 0.12202     | 0.27065     | 0.03767     | 0.07310     | 0.13000     | 0.06945     | 2329.50073  | 0.06938     | 2288.22217   |
|       UKGsE        | no                     | 0.12202     | 0.27065     | 0.03100     | 0.06200     | 0.11300     | 0.06000     | 2973.23600  | 0.06400     | 2641.84000   |
|       UPGAT        | yes                    | 0.02922     | 0.10107     | 0.61800     | 0.75100     | 0.86200     | 0.70100     | 69.12000    | 0.70800     | 69.36400     |
|       UPGAT        | no                     | 0.02922     | 0.10107     | 0.53000     | 0.65400     | 0.76500     | 0.61100     | 115.00400   | 0.65800     | 93.69200     |

<br>


# 🛠️ Deployment

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

**Step3** Model training.
```
python main.py
```

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
<br>


# ✉️ Citation

If you find unKR is useful for your research, please consider citing the following paper:

```bibtex
@article{
}

```
<br>

# 😊 unKR Core Team

**Southeast University**: Jingting Wang, Tianxing Wu, Shilin Chen, Yunchang Liu, Shutong Zhu, Wei Li, Jingyi Xu, Guilin Qi.

# 🔎 Reference
- NeuralKG: An Open Source Library for Diverse Representation
Learning of Knowledge Graphs. SIGIR 2022 Demo. https://arxiv.org/pdf/2202.12571.pdf
- NeuralKG-ind: A Python Library for Inductive Knowledge
Graph Representation Learning. SIGIR 2023 Demo. https://arxiv.org/pdf/2304.14678.pdf
- GitHub：https://github.com/zjukg/NeuralKG

