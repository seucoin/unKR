# -*- coding: utf-8 -*-c
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from unKR.utils import *
from unKR.data.Sampler import *


def main(arg_path):
    args = setup_parser()  # set parser
    args = load_config(args, arg_path)
    seed_everything(args.seed)
    monitor_name = "Eval_" + args.monitor
    model_checkpoint_filename = "{epoch}-{" + monitor_name + ":.5f}"
    if args.monitor == "MAE" or args.monitor == "wmr" or args.monitor == "MSE":
        model_checkpoint_mode = "min"
    else:
        model_checkpoint_mode = "max"

    """set up sampler to datapreprocess"""  # Set up the sampling process for data processing
    train_sampler_class = import_class(f"unKR.data.{args.train_sampler_class}")
    train_sampler = train_sampler_class(args)
    test_sampler_class = import_class(f"unKR.data.{args.test_sampler_class}")
    test_sampler = test_sampler_class(train_sampler)

    """set up datamodule"""
    data_class = import_class(f"unKR.data.{args.data_class}")
    kgdata = data_class(args, train_sampler, test_sampler)

    """set up model"""
    model_class = import_class(f"unKR.model.{args.model_name}")

    if args.model_name == "GMUC" or args.model_name == "GMUCp":
        model = model_class(args, args.num_symbols, None)
    else:
        model = model_class(args)

    """set up lit_model"""
    litmodel_class = import_class(f"unKR.lit_model.{args.litmodel_name}")
    lit_model = litmodel_class(model, train_sampler, args)

    """set up logger"""
    logger = pl.loggers.TensorBoardLogger("training/logs")
    if args.use_wandb:
        log_name = "_".join([args.model_name, args.dataset_name, str(args.lr)])
        logger = pl.loggers.WandbLogger(name=log_name, project="unKR")
        logger.log_hyperparams(vars(args))

    """early stopping"""
    early_callback = pl.callbacks.EarlyStopping(
        monitor=monitor_name,
        mode="max",
        patience=args.early_stop_patience,
        # verbose=True,
        check_on_train_epoch_end=False,
    )

    """set up model save method"""
    # The path where the model is saved
    dirpath = "/".join(["output", args.eval_task, args.dataset_name, args.model_name])
    model_checkpoint = pl.callbacks.ModelCheckpoint(
        monitor=monitor_name,
        mode=model_checkpoint_mode,
        filename=model_checkpoint_filename,
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
    )
    '''保存参数到config'''
    if args.save_config:
        save_config(args)

    if not args.test_only:
        # train&valid
        trainer.fit(lit_model, datamodule=kgdata)
        # Load the model with the best performance on dev in this experiment and test it
        path = model_checkpoint.best_model_path
    else:
        path = args.checkpoint_dir
    lit_model.load_state_dict(torch.load(path)["state_dict"])
    lit_model.eval()
    trainer.test(lit_model, datamodule=kgdata)


if __name__ == "__main__":
    main(arg_path='config/nl27k/GMUC_nl27k.yaml')
