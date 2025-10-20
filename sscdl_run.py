# import sys
# import os
# script_dir = os.path.dirname(os.path.abspath(__file__))
# src_dir = os.path.join(script_dir, 'src')
# sys.path.insert(0, src_dir)
import random
import numpy as np
import torch
torch.set_num_threads(30)
import pytorch_lightning as pl
import torch.autograd
from pytorch_lightning import seed_everything
from unKR.utils import *
from unKR.data.Sampler import *
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


torch.autograd.set_detect_anomaly(True)
def main():
    with torch.autograd.set_detect_anomaly(True):
        parser = setup_parser()  # set parser
        args = parser.parse_args()
        if args.load_config:
            args = load_config(args, args.config_path)
        seed_everything(args.seed)

        """set up sampler to datapreprocess"""
        train_sampler_class = import_class(f"unKR.data.{args.train_sampler_class}")
        train_sampler = train_sampler_class(args)

        test_sampler_class = import_class(f"unKR.data.{args.test_sampler_class}")
        test_sampler = test_sampler_class(train_sampler)

        """set up datamodule"""
        data_class = import_class(f"unKR.data.{args.data_class}")
        kgdata = data_class(args, train_sampler, test_sampler)

        """set up model"""
        model_class = import_class(f"unKR.model.{args.model_name}")
        # model = FUKGCMetaModel(args)
        # model = model_class(args)
        model_meta = model_class(args)
        # model_meta = FUKGC(args)

        """set up lit_model"""
        litmodel_class = import_class(f"unKR.lit_model.{args.litmodel_name}")
        lit_model = litmodel_class(model_meta, args)
        # lit_model = MetaFUKGCLitModel(model_meta, args)
        print(lit_model)

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
        # save model with the best results saved on the validation set
        dirpath = "/".join(["output", args.eval_task, args.dataset_name, args.model_name,''])

        model_checkpoint = pl.callbacks.ModelCheckpoint(
            monitor="Eval_MAE",
            mode="min",
            filename="{epoch}-{Eval_MAE:.5f}",
            dirpath=dirpath,
            save_weights_only=True,
            save_top_k=1,
        )
        model_checkpoint1 = pl.callbacks.ModelCheckpoint(
            monitor="Eval_wmrr",
            mode="max",
            filename="{epoch}-{Eval_wmrr:.5f}",
            dirpath=dirpath,
            save_weights_only=True,
            save_top_k=1,
        )
        model_checkpoint2 = pl.callbacks.ModelCheckpoint(
            monitor="Eval_wmr",
            mode="min",
            filename="{epoch}-{Eval_wmr:.5f}",
            dirpath=dirpath,
            save_weights_only=True,
            save_top_k=1,
        )
        model_checkpoint3 = pl.callbacks.ModelCheckpoint(
            monitor="Eval_MSE",
            mode="min",
            filename="{epoch}-{Eval_MSE:.5f}",
            dirpath=dirpath,
            save_weights_only=True,
            save_top_k=1,
        )
        model_checkpoint_last = pl.callbacks.ModelCheckpoint(
            filename="{epoch}",
            dirpath=dirpath,
            save_weights_only=True,
            save_top_k=-1,
            every_n_epochs=10,
            save_last=True,  # Additionally, always save the model at the last epoch
        )
        # callbacks = [model_checkpoint, model_checkpoint1, model_checkpoint2, model_checkpoint3]
        callbacks = [model_checkpoint, model_checkpoint1, model_checkpoint2, model_checkpoint3, model_checkpoint_last]

        # initialize trainer
        trainer = pl.Trainer.from_argparse_args(
            args,
            callbacks=callbacks,
            logger=logger,
            default_root_dir="training/logs",
            gpus="0,",
            check_val_every_n_epoch=args.check_val_every_n_epoch,
            max_epochs=args.max_epochs,
        )
        '''save parameters to config'''
        if args.save_config:
            save_config(args)

        if not args.test_only:
            # train&valid
            trainer.fit(lit_model, datamodule=kgdata)
            path = model_checkpoint_last.best_model_path
        else:
            path = ""


        lit_model.load_state_dict(torch.load(path)["state_dict"])
        lit_model.eval()
        trainer.test(lit_model, datamodule=kgdata)


if __name__ == "__main__":
    main()
