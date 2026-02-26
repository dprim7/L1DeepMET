import tensorflow as tf # type: ignore
from typing import Any, Dict, Optional, Union
import keras # type: ignore
from keras.models import Model #type: ignore
from keras.callbacks import ( #type: ignore
    Callback,
    History,
    EarlyStopping,
    LearningRateScheduler,
    ReduceLROnPlateau,
    CSVLogger,
    ModelCheckpoint,
    TensorBoard
    )
from keras.optimizers import schedules #type: ignore

import os
from l1deepmet.config import Config


def _cfg_get(config: Union[Config, dict], key: str, default=None):
    """Get a config value from either a Config object (dot-notation) or a plain dict."""
    if isinstance(config, Config):
        return config.get(key, default)
    # plain dict: support one level of dot-notation
    parts = key.split(".", 1)
    val = config.get(parts[0], default if len(parts) == 1 else {})
    if len(parts) == 2 and isinstance(val, dict):
        return val.get(parts[1], default)
    return val if len(parts) == 1 else default


def split_preprocessed_features(X):
    """Split preprocessed H5 features into model input format"""
    inputs = X[:, :, 0:5]      # pt, eta, phi, puppi, hcal_depth
    pxpy = X[:, :, 5:7]        # px, py
    inputs_cat0 = X[:, :, 7]   # encoded_pdgId
    inputs_cat1 = X[:, :, 8]   # encoded_charge
    return inputs, pxpy, inputs_cat0, inputs_cat1


def train_full_precision(
        model: Model,
        config: Union[Config, dict],
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        optimizer: tf.keras.optimizers.Optimizer,
        loss: tf.keras.losses.Loss,
        metrics: list[tf.keras.metrics.Metric],
        batch_size: int = 256,
        path_out: str = "outputs/models/run_0",
    ) -> tuple[History, Model]:

    os.makedirs(path_out, exist_ok=True)
    callbacks = create_callbacks(config, path_out=path_out, batch_size=batch_size)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    training_cfg = _cfg_get(config, "training", {})
    epochs = training_cfg.get("epochs", 100) if isinstance(training_cfg, dict) else 100
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, callbacks=callbacks)

    #TODO:save model

    return history, model


def train_hgq(model: keras.Model, config: Config):
    

    
    
    pass

def create_callbacks(
        config: Union[Config, dict],
        path_out: str,
        batch_size: int
):
    callbacks = []

    early_stopping_config = _cfg_get(config, "callbacks.early_stopping", {}) or {}
    early_stopping = EarlyStopping(
        monitor=early_stopping_config.get("monitor", "val_loss"),
        patience=early_stopping_config.get("patience", 40),
        verbose=1,
        restore_best_weights=False,
    )
    callbacks.append(early_stopping)

    lr_config = _cfg_get(config, "callbacks.learning_rate", {}) or {}
    lr_type = lr_config.get("type")

    if lr_type == "cosine_decay":
        initial_lr = lr_config.get("initial_learning_rate")
        decay_steps = lr_config.get("decay_steps")
        alpha = lr_config.get("alpha")
        warmup_target = lr_config.get("warmup_target")
        warmup_steps = lr_config.get("warmup_steps")

        cosine_decay_schedule = schedules.CosineDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            alpha=alpha,
            name="CosineDecay",
            warmup_target=warmup_target,
            warmup_steps=warmup_steps,
        )

        lr_callback = LearningRateScheduler(
            cosine_decay_schedule,
            verbose=1
        )
        callbacks.append(lr_callback)

    elif lr_type == "cyclical":
        # TODO: implement CyclicLR when the dependency is available
        clr_config = _cfg_get(config, "callbacks.cyclical_lr", {}) or {}
        # clr = CyclicLR(base_lr=..., max_lr=..., step_size=..., mode=...)
        # callbacks.append(clr)
        pass
    else: # reduce learning rate on plateau by default
        reduce_lr_on_plateau = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            verbose=1,
            mode="auto",
            min_delta=0.0001,
            cooldown=0,
            min_lr=0,
        )
        callbacks.append(reduce_lr_on_plateau)

    # Terminate on NaN
    stop_on_nan = tf.keras.callbacks.TerminateOnNaN()
    callbacks.append(stop_on_nan)

    # CSV Logger
    csv_logger = CSVLogger(os.path.join(path_out, "loss_history.log")) #TODO: use pathlib
    callbacks.append(csv_logger)

    # Model Checkpoint
    model_checkpoint = ModelCheckpoint(
        os.path.join(path_out, "model.keras"),
        monitor="val_loss",
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
        save_freq='epoch',
        initial_value_threshold=None
    )
    callbacks.append(model_checkpoint)

    # TensorBoard
    tensorboard = TensorBoard(
        log_dir=os.path.join(path_out, "tensorboard_logs"), #TODO: use pathlib
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        update_freq="batch",
        profile_batch=0,
        embeddings_freq=0,
        write_steps_per_second=True,
    )
    callbacks.append(tensorboard)

    if _cfg_get(config, "pruning.prune"): # check if needed with HGQ
        callbacks.extend(
            get_pruning_callbacks(path_out, _cfg_get(config, "pruning.target_sparsity"))
        )

    return callbacks


