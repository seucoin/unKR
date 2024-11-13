# -*- coding: utf-8 -*-c
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from unKR.utils import *
from unKR.data.Sampler import *


def main(arg_path):
    print('This demo is for testing PASSLEAF')
    args = setup_parser()  # set parameters
    args = load_config(args, arg_path)
    seed_everything(args.seed)
    print(args.dataset_name)

    """set up sampler to datapreprocess"""
    train_sampler_class = import_class(f"unKR.data.{args.train_sampler_class}")
    train_sampler = train_sampler_class(args)  # 这个sampler是可选择的

    test_sampler_class = import_class(f"unKR.data.{args.test_sampler_class}")
    test_sampler = test_sampler_class(train_sampler)

    """set up datamodule"""
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
        monitor="Eval_MAE",
        mode="min",
        patience=args.early_stop_patience,
        # verbose=True,
        check_on_train_epoch_end=False,
    )

    """set up model save method"""
    # save model with the best results saved on the validation set (It is set to MAE by default, and the result type can be replaced by user)
    dirpath = "/".join(["output", args.eval_task, args.dataset_name, args.model_name])
    model_checkpoint = pl.callbacks.ModelCheckpoint(
        monitor="Eval_MAE",
        mode="min",
        filename="{epoch}-{Eval_MAE:.5f}",
        dirpath=dirpath,
        save_weights_only=True,
        save_top_k=1,
    )
    callbacks = [early_callback, model_checkpoint]
    # initialize trainer
    if args.gpu != "cpu":
        trainer = pl.Trainer.from_argparse_args(
            args,
            callbacks=callbacks,
            logger=logger,
            default_root_dir="training/logs",
            gpus="0,",
            check_val_every_n_epoch=args.check_val_every_n_epoch,
            max_epochs=args.max_epochs,  # 添加 max_epochs 参数
        )
    else:
        trainer = pl.Trainer.from_argparse_args(
            args,
            callbacks=callbacks,
            logger=logger,
            default_root_dir="training/logs",
            # gpus="0,",
            check_val_every_n_epoch=args.check_val_every_n_epoch,
            max_epochs=args.max_epochs,  # 添加 max_epochs 参数
        )
    '''save parameters to config'''
    if args.save_config:
        save_config(args)

    if not args.test_only:
        # train&valid
        trainer.fit(lit_model, datamodule=kgdata)
        path = model_checkpoint.best_model_path
    else:
        path = "./output/confidence_prediction/ppi5k/PASSLEAF/epoch=62-Eval_wmr=2.92500.ckpt"
    lit_model.load_state_dict(torch.load(path)["state_dict"])
    lit_model.eval()
    trainer.test(lit_model, datamodule=kgdata)


if __name__ == "__main__":
    main(arg_path='config/PASSLEAFdemo.yaml')
