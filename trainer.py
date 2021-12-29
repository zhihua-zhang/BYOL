import os
import numpy as np
import torch

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import callbacks as pl_callbacks
from pytorch_lightning import seed_everything

from models import BYOL
from utils import accuracy


def supervise_train(model, train_loader, val_loader, args, with_pretrain=False):
    # tensorboard logger
    config_name = f"with_pretrain_ft{args.ft_ratio}" if with_pretrain else "without_pretrain"
    tb_logger = pl_loggers.TensorBoardLogger("./clf_logs/", name=config_name)

    trainer = pl.Trainer(max_epochs=25,
                         gpus=1 if torch.cuda.is_available() else 0,
                         amp_backend="apex",
                         enable_model_summary=False,
                         logger=tb_logger,
                         checkpoint_callback=False)

    trainer.fit(model, train_loader, val_loader)

    model.cuda()
    acc = sum([accuracy(model(x.cuda()), y.cuda())
              for x, y in val_loader]) / len(val_loader)
    if with_pretrain:
        print(
            f"Validation Accuracy with Self-Supervised Pre-training: {acc:.3f}")
        dirs = "clf_results_with_byol"
    else:
        print(
            f"Validation Accuracy without Self-Supervised Pre-training: {acc:.3f}")
        dirs = "clf_results_without_byol"

    os.makedirs(dirs, exist_ok=True)
    np.save(dirs + '/train_loss.npy', model.train_losses)
    np.save(dirs + '/val_loss.npy', model.val_losses[1:])
    np.save(dirs + '/train_acc.npy', model.train_accuracies)
    np.save(dirs + '/val_acc.npy', model.val_accuracies[1:])
    # save the model
    torch.save(model.state_dict(), dirs + '/model_state.pth')


def BYOL_pretrain(model, train_loader, val_loader, args, target_update_rule=(lambda _: 0.999)):
    byol = BYOL(model, target_update_rule,
                args, image_size=args.image_size)

    # reproducibility
    seed_everything(2021, workers=True)

    use_gpu = torch.cuda.is_available()
    config_name = f""

    # model checkpoint
    checkpoint_callback = pl_callbacks.ModelCheckpoint(
        dirpath="./checkpoints",
        filename=config_name + "{epoch}-{val_loss:.3f}-{val_accuracy:.3f}",
        monitor="val_accuracy",
        mode="max",
        every_n_epochs=10)

    # tensorboard logger
    tb_logger = pl_loggers.TensorBoardLogger("./byol_logs/", name=config_name)

    # pl trainer
    trainer = pl.Trainer(
        max_epochs=args.byol_epochs,
        gpus=1 if use_gpu else 0,
        amp_backend="apex",
        accumulate_grad_batches=2048 // args.bz,
        enable_model_summary=False,
        num_sanity_val_steps=-1,
        resume_from_checkpoint=(
            "./checkpoints/" + args.model_pth) if args.model_pth else None,
        callbacks=[checkpoint_callback],
        logger=tb_logger,
    )
    trainer.fit(byol, train_loader, val_loader)

    dirs = "results_byol"
    os.makedirs(dirs, exist_ok=True)
    np.save(dirs + '/byol_train_loss.npy', byol.train_losses)
    np.save(dirs + '/byol_val_loss.npy', byol.val_losses[1:])
    np.save(dirs + '/byol_train_acc.npy', byol.train_accuracies[1:])
    np.save(dirs + '/byol_val_acc.npy', byol.val_accuracies[1:])
    torch.save(model.state_dict(), 'pretrained.pth')

    return byol
