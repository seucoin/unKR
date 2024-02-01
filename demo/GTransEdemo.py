# -*- coding: utf-8 -*-c
# from torch._C import T
# from train import Trainer
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from unKR.utils import *
from unKR.data.Sampler import *
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'


def main(arg_path):
    print('This demo is for testing GTransE')
    args = setup_parser()  # set parameter
    args = load_config(args, arg_path)
    seed_everything(args.seed)
    print(args.dataset_name)

    """set up sampler to datapreprocess"""  
    train_sampler_class = import_class(f"src.unKR.data.{args.train_sampler_class}")
    train_sampler = train_sampler_class(args)  # Optional
    # print(train_sampler)
    test_sampler_class = import_class(f"src.unKR.data.{args.test_sampler_class}")
    test_sampler = test_sampler_class(train_sampler)  # must

    """set up datamodule"""  
    data_class = import_class(f"src.unKR.data.{args.data_class}")
    kgdata = data_class(args, train_sampler, test_sampler)

    """set up model"""
    model_class = import_class(f"src.unKR.model.{args.model_name}")
    model = model_class(args)

    """set up lit_model"""
    litmodel_class = import_class(f"src.unKR.lit_model.{args.litmodel_name}")
    lit_model = litmodel_class(model, args)

    """set up logger"""
    logger = pl.loggers.TensorBoardLogger("training/logs")
    if args.use_wandb:
        log_name = "_".join([args.model_name, args.dataset_name, str(args.lr)])
        logger = pl.loggers.WandbLogger(name=log_name, project="unKR")
        logger.log_hyperparams(vars(args))

    """early stopping"""
    early_callback = pl.callbacks.EarlyStopping(
        monitor="Eval_mrr",
        mode="max",
        patience=args.early_stop_patience,
        # verbose=True,
        check_on_train_epoch_end=False,
    )

    """set up model save method"""
    dirpath = "/".join(["output", args.eval_task, args.dataset_name, args.model_name])
    model_checkpoint = pl.callbacks.ModelCheckpoint(
        monitor="Eval_mrr",
        mode="max",
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
        path = "./output/confidence_prediction/nl27k/GTransE/epoch=13-Eval_MSE=21.55200.ckpt"
    lit_model.load_state_dict(torch.load(path)["state_dict"])
    lit_model.eval()
    trainer.test(lit_model, datamodule=kgdata)


if __name__ == "__main__":
    main(arg_path='../config/nl27k/GTransE_nl27k.yaml')
