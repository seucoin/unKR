# -*- coding: utf-8 -*-c
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from unKR.utils import *
from unKR.data.Sampler import *
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def main(arg_path):
    print('This demo is for testing UKGsE')
    args = setup_parser()  # set parameter
    args = load_config(args, arg_path)
    seed_everything(args.seed)
    print(args.dataset_name)

    """set up sampler to datapreprocess"""  # Set up the sampling process for data processing
    train_sampler_class = import_class(f"unKR.data.{args.train_sampler_class}")
    train_sampler = train_sampler_class(args)  # This sampler is optional
    # print(train_sampler)
    test_sampler_class = import_class(f"unKR.data.{args.test_sampler_class}")
    test_sampler = test_sampler_class(train_sampler)  # test_sampler is a must

    """set up datamodule"""  # set data module
    data_class = import_class(f"unKR.data.{args.data_class}")  
    kgdata = data_class(args, train_sampler, test_sampler)

    """set up model"""
    model_class = import_class(f"unKR.model.{args.model_name}")
    model = model_class(args)

    """set up lit_model"""
    litmodel_class = import_class(f"unKR.lit_model.{args.litmodel_name}")
    lit_model = litmodel_class(model, args)

    """set up logger"""
    logger = pl.loggers.TensorBoardLogger("training/logs")
    if args.use_wandb:
        log_name = "_".join([args.model_name, args.dataset_name, str(args.lr)])
        logger = pl.loggers.WandbLogger(name=log_name, project="unKR")
        logger.log_hyperparams(vars(args))

    """early stopping"""
    early_callback = pl.callbacks.EarlyStopping(
        monitor="Eval_MSE",
        mode="min",
        patience=args.early_stop_patience,
        # verbose=True,
        check_on_train_epoch_end=False,
    )

    """set up model save method"""
    # It is the model with the best MSE results saved on the validation set
    # The path where the model is saved
    dirpath = "/".join(["output", args.eval_task, args.dataset_name, args.model_name])
    model_checkpoint = pl.callbacks.ModelCheckpoint(
        monitor="Eval_MSE",
        mode="min",
        filename="{epoch}-{Eval_MSE:.5f}",
        dirpath=dirpath,
        save_weights_only=True,
        save_top_k=1,
    )
    callbacks = [early_callback, model_checkpoint]
    # initialize trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=callbacks,
        logger=logger,
        default_root_dir="training/logs",
        gpus="0,",
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        max_epochs=args.max_epochs,  # 添加 max_epochs 参数
    )
    '''Save the parameters to config'''
    if args.save_config:
        save_config(args)

    if not args.test_only:
        # train&valid
        trainer.fit(lit_model, datamodule=kgdata)
        # Load the model with the best performance on dev in this experiment and test it
        path = model_checkpoint.best_model_path
    else:
        # path = args.checkpoint_dir
        path = "./output/confidence_prediction/cn15k/UKGsE/epoch=5-Eval_MSE=0.10300.ckpt"
    lit_model.load_state_dict(torch.load(path)["state_dict"])
    lit_model.eval()
    trainer.test(lit_model, datamodule=kgdata)


if __name__ == "__main__":
    main(arg_path='config/cn15k/UKGsE_cn15k.yaml')
