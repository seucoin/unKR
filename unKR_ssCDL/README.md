# Uncertain Knowledge Graph Completion via Semi-Supervised Confidence Distribution Learning

Source code for NeurIPS paper: Uncertain Knowledge Graph Completion via Semi-Supervised Confidence Distribution Learning.

Uncertain knowledge graphs (UKGs) associate each triple with a conffidence score to provide more precise knowledge representations. Recently, since real-world
UKGs suffer from the incompleteness, uncertain knowledge graph (UKG) completion
 attracts more attention, aiming to complete missing triples and confidences.
Current studies attempt to learn UKG embeddings to solve this problem, but
they neglect the extremely imbalanced distributions of triple confidences. This
causes that the learnt embeddings are insufficient to high-quality UKG completion.
Thus, in this paper, to address the above issue, we propose a new **s**emi-**s**upervised
**C**onfidence **D**istribution **L**earning (**ssCDL**) method for UKG completion, where
each triple confidence is transformed into a confidence distribution to introduce
more supervision information of different confidences to reinforce the embedding
learning process. ssCDL iteratively learns UKG embedding by relational learning
on labeled data (i.e., existing triples with confidences) and unlabeled data with
pseudo labels (i.e., unseen triples with the generated confidences), which are predicted
 by meta-learning to augment the training data and rebalance the distribution
of triple confidences. Experiments on two UKG datasets demonstrate that ssCDL
consistently outperforms state-of-the-art baselines in different evaluation metrics.

![ssCDL.png](..%2Fpics%2FssCDL.png)
## Usage

### Installation

**Step1** Create a virtual environment using ```Anaconda``` and enter it.

```bash
conda create -n unKR python=3.8
conda activate unKR
```

**Step2**  Install package.
+ Install from source
```bash
git clone https://github.com/seucoin/unKR.git
cd unKR
pip install -r requirements.txt
python setup.py install
```
+ Install by pypi
```bash
pip install unKR
```

### Data Format
For normal models, `train.tsv`, `val.tsv`, and `test.tsv` are required. 

- `train.tsv`: All facts used for training in the format `(h, r, t, s)`, one fact per line.

- `val.tsv`: All facts used for validation in the format `(h, r, t, s)`, one fact per line.

- `test.tsv`: All facts used for testing in the format `(h, r, t, s)`, one fact per line.

### Model Training
Training on NL27k
```bash
python sscdl_run.py --load_config --config_path ./config/nl27k/ssCDL_nl27k.yaml
```

Training on CN15k
```bash
python sscdl_run.py --load_config --config_path ./config/cn15k/ssCDL_cn15k.yaml
```

