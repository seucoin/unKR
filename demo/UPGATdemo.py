# -*- coding: utf-8 -*-c
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from unKR.utils import *
from unKR.data.Sampler import *


def main(arg_path):
    print('This demo is for testing UPGAT')
    args = setup_parser()  # set parameters
    args = load_config(args, arg_path)
    seed_everything(args.seed)
    print(args.dataset_name)

    if not args.test_only and args.teacher_model:
        print("--------------------------------")
        print(f"Teacher model has been started.")
        """set up sampler to datapreprocess"""
        train_sampler_class = import_class(f"src.unKR.data.{args.train_sampler_class}")
        train_sampler = train_sampler_class(args)
        test_sampler_class = import_class(f"src.unKR.data.{args.test_sampler_class}")
        test_sampler = test_sampler_class(train_sampler)

        """set up datamodule"""
        data_class = import_class(f"src.unKR.data.{args.data_class}")  # define DataClass
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
            monitor="Eval_MAE",   # adaptation to different datasets
            mode="min",
            patience=args.early_stop_patience,
            # verbose=True,
            check_on_train_epoch_end=False,
        )

        """set up model save method"""
        # save the teacher model with the best results saved on the validation set (default to MAE, WMR(ppi5k)/MSE(nl27k)/WMR(cn15k))
        dirpath = "/".join(["output", args.eval_task, args.dataset_name, args.model_name, "teacher_model"])
        model_checkpoint = pl.callbacks.ModelCheckpoint(
            monitor="Eval_MAE",   # adaptation to different datasets
            mode="min",
            filename="{epoch}-{Eval_MAE:.5f}",
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
        '''save parameters to config'''
        if args.save_config:
            save_config(args)

        # train&valid
        trainer.fit(lit_model, datamodule=kgdata)
        # load the best performing model in the experiment for test
        path = model_checkpoint.best_model_path
        lit_model.load_state_dict(torch.load(path)["state_dict"])
        lit_model.eval()
        trainer.test(lit_model, datamodule=kgdata)

        print(f"Teacher model has been trained.")
        print("--------------------------------")
        # generate pseudo triples
        model = model.to(args.gpu)
        train_triples = train_sampler.train_triples
        output_file_path = "/".join([args.data_path, "pseudo.tsv"])
        model.pseudo_tail_predict(train_triples, output_file_path)
        print(f"Pseudo data has been written to {output_file_path}.")
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    print("--------------------------------")
    print(f"Student model has been started.")
    args.teacher_model = False

    """set up sampler to datapreprocess"""
    train_sampler_class = import_class(f"src.unKR.data.{args.train_sampler_class}")
    train_sampler = train_sampler_class(args)
    test_sampler_class = import_class(f"src.unKR.data.{args.test_sampler_class}")
    test_sampler = test_sampler_class(train_sampler)

    """set up datamodule"""
    data_class = import_class(f"src.unKR.data.{args.data_class}")  # define DataClass
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
        monitor="Eval_MAE",   # adaptation to different datasets
        mode="min",
        patience=args.early_stop_patience,
        # verbose=True,
        check_on_train_epoch_end=False,
    )

    """set up model save method"""
    # save the student model with the best results saved on the validation set (default to MAE, WMR(ppi5k)/MSE(nl27k)/WMR(cn15k))
    dirpath = "/".join(["output", args.eval_task, args.dataset_name, args.model_name, "student_model"])
    model_checkpoint = pl.callbacks.ModelCheckpoint(
        monitor="Eval_wmr",   # adaptation to different datasets
        mode="min",
        filename="{epoch}-{Eval_wmr:.5f}",
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
    '''save parameters to config'''
    if args.save_config:
        save_config(args)

    if not args.test_only:
        # train&valid
        trainer.fit(lit_model, datamodule=kgdata)
        # load the best performing model in the experiment for test
        path = model_checkpoint.best_model_path
    else:
        # path = args.checkpoint_dir
        path = "./output/confidence_prediction/ppi5k/UPGAT/student_model/epoch=85-Eval_wmr=2.38164.ckpt"  # adaptation to different datasets
    lit_model.load_state_dict(torch.load(path)["state_dict"])
    lit_model.eval()
    trainer.test(lit_model, datamodule=kgdata)


if __name__ == "__main__":
    main(arg_path='../config/ppi5k/UPGAT_ppi5k.yaml')  # adaptation to different datasets