from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class EarlyStoppingMinEpochs(EarlyStopping):
    def __init__(
        self,
        min_epochs,
        monitor,
        patience,
        mode,
        min_delta=0.0,
        verbose=False,
        strict=True,
    ):
        super().__init__(
            monitor=monitor,
            patience=patience,
            mode=mode,
            min_delta=min_delta,
            verbose=verbose,
            strict=strict,
        )
        self.min_epochs = min_epochs

    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch < self.min_epochs - 1:
            return
        super().on_validation_end(trainer, pl_module)


class ModelCheckpointMinEpochs(ModelCheckpoint):
    def __init__(
        self,
        min_epochs,
        monitor,
        mode,
        dirpath=None,
        filename=None,
        verbose=False,
        save_last=None,
        save_top_k=None,
        save_weights_only=False,
        period=1,
        prefix="",
    ):
        super().__init__(
            dirpath=dirpath,
            filename=filename,
            monitor=monitor,
            verbose=verbose,
            save_last=save_last,
            save_top_k=save_top_k,
            save_weights_only=save_weights_only,
            mode=mode,
            period=period,
            prefix=prefix,
        )
        self.min_epochs = min_epochs

    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch < self.min_epochs - 1:
            return
        super().on_validation_end(trainer, pl_module)
