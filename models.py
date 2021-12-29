import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda import amp
import pytorch_lightning as pl
from copy import deepcopy
from sklearn.linear_model import LogisticRegression
from utils import default_augmentation, byol_loss_fn, supervised_augmentation, image_transform


def mlp(dim, projection_size=256, hidden_size=4096):
    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size),
    )


class SupervisedLightningModule(pl.LightningModule):
    def __init__(self, model, args, **hparams):
        super().__init__()
        self.model = model
        self.args = args
        self.augment = supervised_augmentation()
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):

        optimizer = torch.optim.Adam
        lr = self.args.base_lr / 128 * self.args.bz
        weight_decay = self.args.weight_decay
        return optimizer(self.parameters(), lr=lr, weight_decay=weight_decay)

    def training_step(self, batch, *_):
        x, y = batch
        x = self.augment(x)
        logit = self.forward(x)
        loss = F.cross_entropy(logit, y)
        pred = torch.argmax(logit, dim=1)
        return {"loss": loss, "label": y, "pred": pred}

    def training_epoch_end(self, outputs):
        train_loss = (sum(x["loss"] for x in outputs) / len(outputs)).item()
        self.log("train_loss", train_loss, prog_bar=True)
        self.train_losses.append(train_loss)

        labels = torch.cat([x["label"] for x in outputs])
        preds = torch.cat([x["pred"] for x in outputs])
        train_acc = torch.mean((preds == labels).float()).item()
        self.train_accuracies.append(train_acc)

    @torch.no_grad()
    def validation_step(self, batch, *_):
        x, y = batch
        x = self.augment(x)
        logit = self.forward(x)
        loss = F.cross_entropy(logit, y)
        pred = torch.argmax(logit, dim=1)
        return {"loss": loss, "label": y, "pred": pred}

    @torch.no_grad()
    def validation_epoch_end(self, outputs):
        val_loss = (sum(x["loss"] for x in outputs) / len(outputs)).item()
        self.log("val_loss", val_loss, prog_bar=True)
        self.val_losses.append(val_loss)

        labels = torch.cat([x["label"] for x in outputs])
        preds = torch.cat([x["pred"] for x in outputs])
        val_acc = torch.mean((preds == labels).float()).item()
        self.val_accuracies.append(val_acc)


class EncoderWrapper(nn.Module):

    def __init__(
        self,
        model,
        projection_size=256,
        hidden_size: int = 4096,
        layer=-2,
    ):
        super().__init__()
        self.model = model
        self.projection_size = projection_size
        self.hidden_size = hidden_size
        self.layer = layer

        self._projector = None
        self._projector_dim = None
        self._encoded = torch.empty(0)
        self._representation = torch.empty(0)
        self._register_hook()

    @property
    def projector(self):
        if self._projector is None:
            self._projector = mlp(
                self._projector_dim, self.projection_size, self.hidden_size
            )
        return self._projector

    def _hook(self, module, input, output):
        output = output.flatten(start_dim=1)
        if self._projector_dim is None:
            self._projector_dim = output.shape[-1]
        self._encoded = self.projector(output)
        self._representation = output

    def _register_hook(self):
        if isinstance(self.layer, str):
            layer = dict([*self.model.named_modules()])[self.layer]
        else:
            # resnet18.layer[-2]: AdaptiveAvgPool2d(output_size=(1, 1))
            # resnet18.layer[-1]: Linear(in_features=512, out_features=1000, bias=True)
            layer = list(self.model.children())[self.layer]

        layer.register_forward_hook(self._hook)

    def forward(self, x):
        _ = self.model(x)
        return self._encoded, self._representation


class BYOL(pl.LightningModule):
    def __init__(
        self,
        model,
        target_update_rule,
        args,
        image_size=(128, 128),
        hidden_layer=-2,
        projection_size=256,
        hidden_size=4096,
        augment_fn=None,
        T_max=50,
    ):
        super().__init__()
        self.augment = default_augmentation(
            image_size) if augment_fn is None else augment_fn
        self.image_transform = image_transform(image_size)
        # rule for updating the target decay rate
        self.target_update_rule = target_update_rule
        self.encoder = EncoderWrapper(
            model, projection_size, hidden_size, layer=hidden_layer
        )
        self.predictor = mlp(projection_size, projection_size, hidden_size)
        self._target = None

        self.encoder(torch.zeros(2, 3, *image_size))

        self.args = args
        self.T_max = T_max
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.clf = LogisticRegression

        self.step_num = 0  # count which training step we're on

    def forward(self, x):
        output, representation = self.encoder(x)
        output = self.predictor(output)
        return output, representation

    @property
    def target(self):
        if self._target is None:
            self._target = deepcopy(self.encoder)
        return self._target

    def update_target(self):
        # update tau based on passed in update rule)
        tau = self.target_update_rule(self.step_num)
        ## debug:
        # if self.step_num % 10 == 0:
        #     print(f'Step {self.step_num} tau = {tau}')
        self.tau = tau
        for p, pt in zip(self.encoder.parameters(), self.target.parameters()):
            pt.data = tau * pt.data + (1 - tau) * p.data

    # --- Methods required for PyTorch Lightning only! ---

    def configure_optimizers(self):
        optimizer = torch.optim.Adam
        # base_lr=1e-4
        lr = self.args.base_lr / 128 * self.args.bz
        weight_decay = self.args.weight_decay
        opt = optimizer(self.parameters(), lr=lr, weight_decay=weight_decay)
        #sched = {"scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.T_max, eta_min=0)}
        # return {"optimizer": opt, "lr_scheduler": sched}
        return opt

    def training_step(self, batch, *_):
        x, _ = batch
        with torch.no_grad():
            x1, x2 = self.augment(x), self.augment(x)

        pred1, _ = self.forward(x1)
        pred2, _ = self.forward(x2)

        with torch.no_grad():
            targ1, _ = self.target(x1)
            targ2, _ = self.target(x2)
        loss = torch.mean(byol_loss_fn(pred1, targ2) +
                          byol_loss_fn(pred2, targ1))

        self.log("train_loss", loss.item())
        self.update_target()
        self.step_num += 1

        return {"loss": loss}

    def training_epoch_end(self, outputs):
        train_loss = (sum(x["loss"] for x in outputs) / len(outputs)).item()
        self.log("train_epoch_loss", train_loss, prog_bar=True)
        self.train_losses.append(train_loss)

        print(f'Current target decay rate: {self.tau}')

    @torch.no_grad()
    def validation_step(self, batch, *_):
        x, label, from_train = batch
        _, rep = self.encoder(self.image_transform(x))
        if from_train[0] and from_train[-1]:
            # For fine-tuning train data, we only return representation
            return {"rep": rep, "label": label}
        elif (not from_train[0]) and (not from_train[-1]):
            # For fine-tuning val data, we also calculate byol val loss
            x1, x2 = self.augment(x), self.augment(x)
            pred1, _ = self.forward(x1)
            pred2, _ = self.forward(x2)
            targ1, _ = self.target(x1)
            targ2, _ = self.target(x2)
            loss = torch.mean(byol_loss_fn(pred1, targ2) +
                              byol_loss_fn(pred2, targ1))
            return {"rep": rep, "label": label, "loss": loss}
        # at most one mini-batch arrives here
        return {}

    def validation_epoch_end(self, outputs):
        train_rep, val_rep = [], []
        train_labels, val_labels = [], []
        val_loss = []
        for out in outputs:
            if "loss" in out:
                # val data
                val_rep.append(out["rep"])
                val_labels.append(out["label"])
                val_loss.append(out["loss"])
            elif "rep" in out:
                # train data
                train_rep.append(out["rep"])
                train_labels.append(out["label"])

        # byol val loss
        val_loss = (sum(val_loss) / (len(val_loss) + 1e-7)).item()
        self.log("val_loss", val_loss)
        self.val_losses.append(val_loss)

        # fine-tuning accuracy
        train_rep, val_rep = map(lambda x: torch.cat(
            x).cpu().detach().numpy(), [train_rep, val_rep])
        train_labels, val_labels = map(lambda x: torch.cat(
            x).cpu().detach().numpy(), [train_labels, val_labels])

        clf = self.clf(max_iter=500, solver="liblinear")
        clf.fit(train_rep, train_labels)

        train_accuracy = clf.score(train_rep, train_labels)
        val_accuracy = clf.score(val_rep, val_labels)

        self.log("train_accuracy", torch.tensor(train_accuracy), prog_bar=True)
        self.log("val_accuracy", torch.tensor(val_accuracy), prog_bar=True)
        self.train_accuracies.append(train_accuracy)
        self.val_accuracies.append(val_accuracy)
