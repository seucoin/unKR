

# Description of basic parameters

`num_layers` : The number of layers in some GNN model.

`decoder_model` : The name of decoder model, in some model.

`eval_task` : The task of validation, default link_prediction.

`calc_hits` : Calculate the hit rate, default [1,3,5,10].

`gpu` : Select the GPU in training, default cuda:0.

`filter_flag` : Filter in negative sampling.

`use_wandb` : Use “weight and bias” to record the result.

`use_weight` : Use subsampling weight.

`checkpoint_dir` : The checkpoint model path.

`save_config` : Save parameters config file.

`load_config` : Load parameters config file.

`config_path` : The config path.

`model_name` : The name of model.

`monitor` : The index name of early stopping.

`dataset_name`: The name of dataset.

`data_path` : The folder path of dataset.

`data_class` : The name of data preprocessing module, default KGDataModule.

`litmodel_name` : The name of processing module of training, evaluation and testing, default KGELitModel.

`train_sampler_class` : Sampling method used in training, default UniSampler.

`test_sampler_class` : Sampling method used in validation and testing, default TestSampler.

`loss_name` : The name of loss function.

`negative_adversarial_sampling` : Use self-adversarial negative sampling.

`optim_name` : The name of optimizer.

`seed`: Random seed.

`margin` : The fixed margin in loss function.

`adv_temp` : The temperature of sampling in self-adversarial negative sampling.

`emb_dim` : The embedding dimension in KGE model.

`out_dim` : The output embedding dimmension in some KGE model.

`max_epochs` : The maximum epoch in training.

`lr` : Learning rate

`train_bs` : Batch size in training.

`eval_bs` : Bathc size in evaluation and testing.

`num_neg` : The number of negative samples corresponding to each positive sample

`num_ent` : The number of entity, autogenerate.

`num_rel` : The number of relation, autogenerate.

`check_per_epoch` : Evaluation per n epoch of training.

`early_stop_patience` : If the number of consecutive bad results is n, early stop.


