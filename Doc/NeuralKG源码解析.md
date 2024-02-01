



# 前言

本文意在分析NeuralKG源码中所提供的demo.py文件，借助该文件所用到的类，将项目中的各模块串联起来，帮助理解。



# 一、参数设置



```python
if __name__ == "__main__":
    main(arg_path='config/TransE_demo_kg.yaml')
```

主函数中调用main函数，并传入格式为yaml的文件，作为参数

该文件中包含了一个模型在一个数据集当中所采用的优化算法、采样器等，也包含迭代轮次，学习率等超参数在内的一系列信息



```python
print('This demo is powered by \033[1;32mNeuralKG \033[0m')
args = setup_parser()  # 批量设置默认的超参数
args = load_config(args, arg_path)  # 读入yaml文件中的参数设置
seed_everything(args.seed)
```



# 二、采样模块



```python
"""set up sampler to datapreprocess"""  # 设置数据处理的采样过程
train_sampler_class = import_class(f"neuralkg.data.{args.train_sampler_class}")
train_sampler = train_sampler_class(args)  # 这个sampler是可选择的
# print(train_sampler)
test_sampler_class = import_class(f"neuralkg.data.{args.test_sampler_class}")
test_sampler = test_sampler_class(train_sampler)  # test_sampler是一定要的
```

这段代码使用了`import_class`方法，从指定路径中读入了对应的`sample`类，作为采样器

回到yaml文件当中我们可以看到，这里导入的是`UniSampler`和`TestSampler`两个类，这里着重记录下`UniSample`，`TestSampler`同理

找到`UniSample`类可以发现，其父类为`BaseSampler`。而`BaseSampler`的父类又是`KGData`，因此从KGData中开始看



## KGData

加载数据，并进行最初始的数据处理

```python
class KGData(object):
    """Data preprocessing of kg data."""

    def __init__(self, args):

    def get_id(self):

    def get_triples_id(self):

    def get_hr2t_rt2h_from_train(self):
       
    @staticmethod
    def count_frequency(triples, start=4):

    def get_h2rt_t2hr_from_train(self):

    def get_hr_trian(self):

```

KGData类中一共有7个方法

- init：定义了实体关系到id的映射字典，三元组列表等数据结构；
- get_id：**文件读入**。读入全部的实体与关系，记录映射关系和实体关系总数；
- get_triples_id：**文件读入**。读入训练集、验证集和测试集，并以三元组id的格式存储；
- get_hr2t_rt2h_from_train：处理三元组，得到hr2t和rt2h格式的数据；
- count_frequency：记录(head, relation)和(relation, tail)的数量；
- get_h2rt_t2hr_from_train：处理三元组，得到h2rt和t2hr格式的数据；
- get_hr_trian：将字典hr2t_train，转为元组的形式(hr, t)，存于列表中



## BaseSampler

`KGData`的子类

```python
class BaseSampler(KGData):
    def __init__(self, args):
        super().__init__(args)
        self.get_hr2t_rt2h_from_train()

    def corrupt_head(self, t, r, num_max=1):
        
    def corrupt_tail(self, h, r, num_max=1):

    def head_batch(self, h, r, t, neg_size=None):

    def tail_batch(self, h, r, t, neg_size=None):

    def get_train(self):
        return self.train_triples

    def get_valid(self):
        return self.valid_triples

    def get_test(self):
        return self.test_triples

    def get_all_true_triples(self):
        return self.all_true_triples
```

共有9个方法：

- init：初始化，并从训练集中生成hr2t和rt2h格式的数据；

- corrupt_head：给定尾实体和关系，替换头实体生成负样本

  ```python
  def corrupt_head(self, t, r, num_max=1):
      # 随机生成num_max个整数，大小在0-self.args.num_ent之间
      tmp = torch.randint(low=0, high=self.args.num_ent, size=(num_max,)).numpy()
      if not self.args.filter_flag:
          return tmp
      # 如果tmp的值在后者中，返回false，否则返回true
      mask = np.in1d(tmp, self.rt2h_train[(r, t)], assume_unique=True, invert=True)
      # 过滤，保留不在原样本中的数据，作为负样本。保证了构造的负样本确实不在数据集中
      neg = tmp[mask]
      return neg
  ```

- corrupt_tail：同corrupt_head，不过变成了替换尾实体；

- head_batch：通过调用corrupt_head方法，生成neg_size数量的头实体负样本；

- tail_batch：同head_batch；

- get_train：返回训练集三元组；

- get_valid：返回验证集三元组；

- get_test：返回测试集三元组；

- get_all_true_triples：返回全部的三元组

## UniSample

`BaseSampler`的子类

```python
class UniSampler(BaseSampler):
    def __init__(self, args):
        super().__init__(args)
        self.cross_sampling_flag = 0

    def sampling(self, data):

    def uni_sampling(self, data):

    def get_sampling_keys(self):
        return ['positive_sample', 'negative_sample', 'mode']
```

共有4个方法：

- **sampling**：输入为data（三元组列表），返回为一个字典，包含了mode（头实体替换or尾实体替换），`positive_sample`正样本，`negative_sample`为头实体负样本or尾实体负样本（不是一个完整的三元组）

  <img src="C:\Users\codeslogan\AppData\Roaming\Typora\typora-user-images\image-20230909165352957.png" alt="image-20230909165352957" style="zoom:50%;" />

- uni_sampling：同sampling，不过可以同时得到头实体与尾实体的负样本



## TestSampler

`TestSampler`会将`TrainSample`训练集的采样器作为对象，进行调用

```python
class TestSampler(object):
    def __init__(self, sampler):

    def get_hr2t_rt2h_from_all(self):

    def sampling(self, data):

    def get_sampling_keys(self):

```

- init：初始化采样器；
- get_hr2t_rt2h_from_all：从数据集当中（训练+验证+测试集），得到格式为hr2t和rt2h的数据，数据类型为tensor
- sampling：输入为data三元组，输出为一个字典，其中包含了positive_sample原始三元组，head_label和tail_label是一个0-1值的二维张量
- get_sampling_keys：batch_data的键



# 三、数据模块

```python
"""set up datamodule""" #设置数据模块
data_class = import_class(f"neuralkg.data.{args.data_class}") #定义数据类 DataClass
kgdata = data_class(args, train_sampler, test_sampler)
```

这段代码使用了`import_class`方法，导入并实例化了指定的数据模块类，`demo.py`这里导入的是`KGDataModule`类，其父类为`BaseDataModule`，而`BaseDataModule`的父类又是`pl`库下的`LightningDataModule`，接下来从父类开始梳理。


## pl.LightningDataModule

LightningDataModule是一种将数据与模型分离的设计，因此这样可以在不知道不用考虑模型的情况下开发数据集。

因为重写了new函数，在创建子类时，会调用子类的setup函数。

## BaseDataModule

```python
class BaseDataModule(pl.LightningDataModule):
    """Base DataModule."""

    def __init__(self, args):

    @staticmethod
    def add_to_argparse(parser):

    def prepare_data(self):

    def setup(self, stage=None):
       
    def train_dataloader(self):

    def val_dataloader(self):

    def test_dataloader(self):

    def get_config(self):

```

BaseDataModule类中一共有7个方法

- init:初始化参数
- add_to_argparse：用于添加一些参数设置
- prepare_data:空函数，子类应该会重写
- setup:空函数，子类应该会重写
- train_dataloader:将Dataset封装成DataLoader,下同
- val_dataloader:
- test_dataloader:
- get_config:返回标签数量

## KGDataModule

```python
class KGDataModule(BaseDataModule):
    """Base DataModule."""

    def __init__(self, args, train_sampler, test_sampler):

    def get_data_config(self):

    def prepare_data(self):

    def setup(self, stage=None):

    def get_train_bs(self):
       
    def train_dataloader(self):

    def val_dataloader(self):

    def test_dataloader(self):
```

KGDataModule类中一共有8个方法

- init:接受所有参数args，接受train_sampler和test_sampler
- get_data_config: 返回数据集的某些重要参数设置
- prepare_data: 
- setup:生成训练集、验证集、测试集，均为torch Dataset类型
- get_train_bs:通过设置的num_batches，来计算批量大小(batch_size = )
- get_train_bs:通过设置的num_batches，来计算批量大小(batch_size = len()//num_batches)

- train_dataloader:将Dataset封装成DataLoader,下同
- val_dataloader:
- test_dataloader:


# 四、模型模块

这里以demo.py文件来分析模型模块的执行流程与功能

## Model

model文件夹下按照功能划分包括许多模块，如KGEModel（知识图谱嵌入），GNNModel（图神经网络），RuleModel等，此处demo.py文件使用的是经典的TransE模型，属于KGEModel模块。

KGEModel和RuleModel两类模型的模块中，都有一个model.py文件，里面有Model类，继承了pytorch中的nn.module，本模块的其他模型之后都会继承该Model类。我们以demo.py里用的KGEModel下的model.py进行分析，该类结构如下，分析见注释：

```python
class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
	#以下几个函数需要在实现具体KGE模型的时候进行重写
    def init_emb(self):
        raise NotImplementedError

    def score_func(self, head_emb, relation_emb, tail_emb):
        raise NotImplementedError

    def forward(self, triples, negs, mode):
        raise NotImplementedError
        
    def tri2emb(self, triples, negs=None, mode="single"):
        #本函数是该模块的核心函数，所有的KGE模型都会基于tri2emb进行嵌入。该函数的作用在于获取三元组的嵌入，该嵌入是三维的（及头实体、关系、尾实体的嵌入）可以通过negs选择被用作负采样的实体，然后mode可以被用来负采样实体的替换方式（比如是替换头实体还是尾实体）。通常情况下，不考虑负采样的情况下，只需要输入三元组参数即可进行嵌入。
        #具体代码较长，见源文件
        return head_emb, relation_emb, tail_emb #返回值为头实体、关系、尾实体的嵌入
```



## TransE

TransE是一个具体的知识嵌入模型，它继承了上文提到的Model类，具体框架如下：

```python
class TransE(Model):
    def __init__(self, args):
        super(TransE, self).__init__(args)
        self.args = args
        self.ent_emb = None
        self.rel_emb = None
        self.init_emb() #调用init_emb函数，进行实体和关系嵌入向量的初始化
        
    def init_emb(self):
        #按照均匀分布对实体和关系的嵌入向量进行初始化，无return值，具体见源码
    
    def score_func(self, head_emb, relation_emb, tail_emb, mode):
        #根据公式 math:`\gamma - ||h + r - t||_F` 计算三元组得分，返回得分值
        score = (head_emb + relation_emb) - tail_emb
        score = self.margin.item() - torch.norm(score, p=1, dim=-1)
        return score
    
    def forward(self, triples, negs=None, mode='single'):
        #重写forward函数，forward函数可以获得三元组的嵌入向量以及三元组得分
        head_emb, relation_emb, tail_emb = self.tri2emb(triples, negs, mode)
        score = self.score_func(head_emb, relation_emb, tail_emb, mode)
        return score
    
    def get_score(self, batch, mode):
        #该函数用于测试环节，获得三元组得分，逻辑和forward一致
        triples = batch["positive_sample"]
        head_emb, relation_emb, tail_emb = self.tri2emb(triples, mode=mode)
        score = self.score_func(head_emb, relation_emb, tail_emb, mode)
        return score
```

在main.py中，通过如下代码，根据设置的参数里的model_name，利用import_class方法构造model_class。然后再实例化一个model作为model_class类的具体模型。demo.py传入的参数的model_name即transE。

```python
model_class = import_class(f"neuralkg.model.{args.model_name}")
    
    if args.model_name == "RugE":
        ground = GroundAllRules(args)
        ground.PropositionalizeRule()

    if args.model_name == "ComplEx_NNE_AER":
        model = model_class(args, train_sampler.rel2id)
    elif args.model_name == "IterE":
        print(f"data.{args.train_sampler_class}")
        model = model_class(args, train_sampler, test_sampler)
    else:
        model = model_class(args)
```

## BaseLitModel

litmodel是基于pytorch lightning构造的一系列模型，所有的litmodel都会继承BaseLitModel进行编写。BaseLitModel继承了pl.LightningModule。BaseLitModel的框架如下：

```python
class BaseLitModel(pl.LightningModule):
    def __init__(self, model, args: argparse.Namespace = None, src_list = None, dst_list=None, rel_list=None):
        super().__init__()
        self.model = model
        self.args = args
        optim_name = args.optim_name
        self.optimizer_class = getattr(torch.optim, optim_name)
        loss_name = args.loss_name
        self.loss_class = getattr(loss, loss_name)
        self.loss = self.loss_class(args, model)
        if self.args.model_name == 'SEGNN':
            self.automatic_optimization = False
            
    def add_to_argparse(parser):
        #为用户自动补充一些指令以设置部分lr和weight_decay等参数的默认值
        parser.add_argument("--lr", type=float, default=0.1)
        parser.add_argument("--weight_decay", type=float, default=0.01)
        return parser
    
    def configure_optimizers(self):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        raise NotImplementedError
    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        raise NotImplementedError

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        raise NotImplementedError
    #以上raise NotImplementedError的函数都需要后续继承该类的LitModel进行重写
    
    def get_results(self, results, mode):
        """Summarize the results of each batch and calculate the final result of the epoch
        Args:
            results ([type]): The results of each batch
            mode ([type]): Eval or Test
        Returns:
            dict: The final result of the epoch
        """
        #获取每轮epoch得到的输出，return值为字典格式
		#代码略
        return outputs
```

demo.py使用下列语句初始化litmodel:

```python
"""set up lit_model"""
litmodel_class = import_class(f"neuralkg.lit_model.{args.litmodel_name}")
lit_model = litmodel_class(model, args)
```

## KGELitModel

transE所对应的litModel即为KGELitModel，TransE等模型都会依托KGELitModel的架构，按照pytorch lightning的规则进行一些重要函数（如training_step、test_step等）的重写。

```python
class KGELitModel(BaseLitModel):
    """Processing of training, evaluation and testing.
    """

    def __init__(self, model, args):
        super().__init__(model, args) #model是通过litmodel_class(model, args)传进来的，所以这里就是transE

    def forward(self, x):
        return self.model(x) #此处forward函数是直接调用了model，所以本质上是直接执行model（这里是transE）的forward函数
    
    # 然后就是按照pl的规则来重写几个函数
    def training_step(self, batch, batch_idx):
        # 训练一步的逻辑
        return loss
    def validation_step(self, batch, batch_idx):
    	# 验证的一步逻辑
        # 进行链接预测任务，获得hit@1,3,10以及mrr指标
    def validation_epoch_end(self, results) -> None:
        # 一步验证结束后的处理
        # 用字典格式记录输出
    def test_step(self, batch, batch_idx):
        # 测试的一步逻辑，与validation_step类似
    def test_epoch_end(self, results) -> None:
        # 一步测试结束后的处理，与validation_epoch_end类似
    def configure_optimizers(self):
        #设置优化器和调度器的参数
```



# 五、监听模块

该模块pytorch-lighting提供了现有的方法，无需重复实现，记录每个参数的方法即可

- pl.loggers.TensorBoardLogger
- pl.callbacks.EarlyStopping
- pl.callbacks.ModelCheckpoint
- pl.Trainer.from_argparse_args（训练器）



  