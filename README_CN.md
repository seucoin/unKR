# unKR：面向不确定性知识图谱表示学习的开源工具包
<p align="center">
    <a href="https://pypi.org/project/unKR/">
        <img alt="Pypi" src="https://img.shields.io/pypi/v/unKR">
    </a>
    <a href="https://github.com/CodeSlogan/unKR/blob/main/LICENSE">
        <img alt="Pypi" src="https://img.shields.io/badge/license-Apache--2.0-yellowgreen">
    </a>
    <!-- <a href="">
        <img alt="LICENSE" src="https://img.shields.io/badge/license-MIT-brightgreen">
    </a> -->
    <a href="https://codeslogan.github.io/unKR/">
        <img alt="Documentation" src="https://img.shields.io/badge/Doc-online-blue">
    </a>
</p>
<p align="center">
    <b> <a href="https://github.com/CodeSlogan/unKR/blob/main/README.md">English</a> | 中文 </b>
</p>

unKR是一个面向不确定性知识图谱表示学习（UKRL）的开源工具包。
其基于[PyTorch Lightning](https://www.pytorchlightning.ai/)框架解耦UKRL模型的工作流程，以实现多种不确定性知识图谱嵌入（Uncertain Knowledge Graph Embedding, UKGE）方法，进而辅助知识图谱补全、推理等工作。
该工具提供了多种已有UKGE模型的代码实现和结果，并为使用者提供了详细的[技术文档](https://codeslogan.github.io/unKR/index.html)。

<br>



# 🔖 概览

<h3 align="center">
    <img src="pics/unKR.svg", width="600">
</h3>
<!-- <p align="center">
    <a href=""> <img src="pics/unKR.svg" width="400"/></a>
<p> -->

unKR工具包是基于[PyTorch Lightning](https://www.pytorchlightning.ai/)框架，用于不确定性知识图谱表示学习的一种高效实现。
它提供了一个可实现多种不确定性知识图谱嵌入模型的模块化流程，包括不确定性知识图谱数据处理模块（负采样模块），模型实现基础模块以及模型训练、验证、测试模块。这些模块被广泛应用于多种UKGE模型中，便于使用者快速构建自己的模型。

已有模型根据是否为小样本场景进行划分，共包含九种不同模型。unKR分别在三种数据集及七种不同的评估指标上完成了工具包有效性验证，模型具体内容在后续部分展开。

unKR的核心开发团队将对该工具包提供长期的技术支持，同时也欢迎开发者们进行探讨研究，可使用 `issue` 发起问题。

关于unKR技术及结果的详细文档请查阅[📋](https://codeslogan.github.io/unKR/)。


<br>

# 💻 运行示例
下面展示了unKR的安装过程，并以[PASSLEAF](http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-rela)模型为例，给出如何进行模型训练以及测试的示例。

[//]: # (<!-- ![框架]&#40;./pics/demo.gif&#41; -->)

[//]: # (<img src="pics/demo.gif">)

[//]: # (<!-- <img src="pics/demo.gif" width="900" height="476" align=center> -->)

<br>

# 📝 模型
unKR实现了九种不确定性知识图谱嵌入方法，根据模型是否为小样本场景进行划分。已有模型如下所示。

|   类型   |                                                                                                                                                                                                                                                   模型                                                                                                                                                                                                                                                   |
|:------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|  一般场景  | [BEURrE](https://aclanthology.org/2021.naacl-main.68)，[FocusE](https://www.ijcai.org/proceedings/2021/395)，[GTransE](https://link.springer.com/chapter/10.1007/978-3-030-39878-1_16)，[PASSLEAF](http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-rela)，[UKGE](https://ojs.aaai.org/index.php/AAAI/article/view/4210)，[UKGsE](https://www.sciencedirect.com/science/article/abs/pii/S0020025522007885)，[UPGAT](https://link.springer.com/chapter/10.1007/978-3-031-33377-4_5) |
| 小样本场景  |                                                                                                                                                                             [GMUC](https://link.springer.com/chapter/10.1007/978-3-030-73194-6_18)，[GMUC+](https://link.springer.com/chapter/10.1007/978-981-19-7596-7_2)                                                                                                                                                                              |




## 数据集
unKR提供了三种不同来源的不确定性知识图谱数据集，包括CN15K、NL27K以及PPI5K。下表展示了三种数据集的来源及其包含的实体数、关系数和四元组数量。

|  数据集  |     来源     |  实体数  | 关系数 |  四元组数  |
|:-----:|:----------:|:-----:|:---:|:------:|
| CN15K | ConceptNet | 15000 | 36  | 241158 |
| NL27K |    NELL    | 27221 | 404 | 175412 |
| PPI5K |   STRING   | 4999  |  7  | 271666 |

对每个数据集进行整理，分别包含以下三个通用于所有模型的数据文件。

`train.tsv`：所有用于训练的三元组及其对应的置信度分数，格式为`(ent1, rel, ent2, score)`，每行表示一个四元组。

`val.tsv`：所有用于验证的三元组及其对应的置信度分数，格式为`(ent1, rel, ent2, score)`，每行表示一个四元组。

`test.tsv`：所有用于测试的三元组及其对应的置信度分数，格式为`(ent1, rel, ent2, score)`，每行表示一个四元组。

在[UKGE](https://ojs.aaai.org/index.php/AAAI/article/view/4210)中，还需使用`softlogic.tsv`文件。

`softloic.tsv`： 所有经由PSL规则推理得到的三元组及其被推理出的置信度分数，格式为`(ent1, rel, ent2, score)`，每行表示一个四元组。

在[GMUC](https://link.springer.com/chapter/10.1007/978-3-030-73194-6_18)，[GMUC+](https://link.springer.com/chapter/10.1007/978-981-19-7596-7_2)中，还需使用以下五个数据文件。

`train/dev/test_tasks.json`：小样本数据集，每一个关系为一个任务，格式为`{rel:[[ent1, rel, ent2, score], ...]}`。字典的键为任务名称，值为该任务下的所有四元组。

`path_graph`：除训练、验证和测试任务外所有数据，即背景知识，格式为`(ent1, rel, ent2, score)`。每行表示一个四元组。

`ontology.csv`：GMUC+模型所需本体知识数据，格式为`(number, h, rel, t)`，每行表示一个本体知识。其中**rel**共有四种，分别为**is_A**，**domain**，**range**，**type**。

- c1 **is_A** c2：c1是c2的**子类**；
- c1 **domain** c2：c1的**定义域**是c2；
- c1 **range** c2：c1的**值域**是c2；
- c1 **type** c2：c1的**类型**是c2。


## 结果
unKR使用了置信度预测和链接预测任务，在MSE、MAE（置信度预测）、Hits@k（k=1,3,10）、MRR、MR、WMRR以及WMR（链接预测）七种不同的指标进行模型评估，并且进行了raw和filter的设置。此外，unKR还采取了高置信度过滤（设置过滤值为0.7）的评估方法。

下面展示了使用unKR的不同模型在NL27K上的结果，更多结果请访问[此处](https://codeslogan.github.io/unKR/result.html)。

### 置信度预测结果
<table>
    <thead>
        <tr>
            <th>类型</th>
            <th>模型</th>
            <th>MSE</th>
            <th>MAE </th>
        </tr>
    </thead>
    <tbody align="center" valign="center">
        <tr>
            <td rowspan="10">一般场景</td>
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
            <td rowspan="2">小样本场景</td>
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

### 链接预测结果（在高置信度测试数据上过滤）
<table>
    <thead>
        <tr>
            <th>类型</th>
            <th>模型</th>
            <th>Hits@1</th>
            <th>Hits@3</th>
            <th>Hits@10</th>
            <th>MRR</th>
            <th>MR</th>
        </tr>
    </thead>
    <tbody align="center" valign="center">
        <tr>
            <td rowspan="12">一般场景</td>
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
            <td rowspan="2">小样本场景</td>
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


# 🛠️ 部署

## 安装

**Step1** 使用 ```Anaconda``` 创建虚拟环境，并进入虚拟环境。

```bash
conda create -n unKR python=3.8
conda activate unKR
pip install -r requirements.txt
```

**Step2** 安装unKR。
+ 基于源码
```bash
git clone https://github.com/CodeSlogan/unKR.git
cd unKR
python setup.py install
```
+ 基于pypi
```bash
pip install unKR
```
## 数据格式
```
All models:
    train/val/test.tsv: (ent1, rel, ent2, score)
UKGE model:
    softloic.tsv: (ent1, rel, ent2, score)
GMUC, GMUC+ models:
    train/dev/test_tasks.json: {rel:[[ent1, rel, ent2, score], ...]}
    path_graph: (ent1, rel, ent2, score)
    ontology.csv: (number, h, rel, t)
```

## 参数调整
在[config](https://github.com/CodeSlogan/unKR/tree/main/config)文件中，unKR提供了复现结果的参数配置文件，具体使用时可以对以下的参数进行调整。

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

## 模型训练
```bash
python main.py --load_config --config_path <your-config>
```

## 模型测试
```bash
python main.py --test_only --checkpoint_dir <your-model-path>
```

## 模型定制
如果您想使用unKR个性化实现自己的模型，需要定义以下的函数/类。

`data`：实现数据处理函数，包括`DataPreprocess`、`Sampler`和`KGDataModule`。
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

`lit_model`：实现模型训练、验证以及测试函数。
```
<your-model-name>LitModel.py:
    class unKR.lit_model.<your-model-name>LitModel.<your-model-name>LitModel
```
`loss`：实现损失函数。
```
<your-model-name>_Loss.py:
    class unKR.loss.<your-model-name>_Loss.<your-model-name>_Loss
```
`model`：实现模型框架函数，根据模型是否为小样本场景分为`UKGModel`和`FSUKGModel`。
```
<your-model-name>.py:
    class unKR.model.UKGModel/FSUKGModel.<your-model-name>.<your-model-name>
```
`config`：实现参数设置。
```
<your-model-name>_<dataset-name>.yaml:
    data_class, litmodel_name, loss_name, model_name, test_sampler_class, train_sampler_class
```
`demo`：实现模型运行文件。
```
<your-model-name>demo.py
```
<br>



# 😊 unKR核心团队

**东南大学**: 王靖婷，吴天星，陈仕林，刘云畅，朱曙曈，李伟，许婧怡，漆桂林。

