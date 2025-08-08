import tensorflow as tf # type: ignore
from typing import Any, Dict, Optional, Config
import keras # type: ignore
from config import Config

def train_hgq(model: keras.Model, config: Config):
    

    
    
    pass

def create_callbacks(
        config: Config, 
        path_out: str, 
        samples_size: int, 
        batch_size: int
):
    callbacks = []

    early_stopping_config = config.get("callbacks.early_stopping", {})
    early_stopping = EarlyStopping(
        monitor=early_stopping_config.get("monitor", "val_loss"),
        patience=early_stopping_config.get("patience", 40),
        verbose=1,
        restore_best_weights=False,
    )
    callbacks.append(early_stopping)

    lr_config = config.get("callbacks.learning_rate", {})
    lr_type = lr_config.get("type")

    if lr_type == "cosine_decay":
        initial_lr = lr_config.get("initial_learning_rate")
        decay_steps = lr_config.get("decay_steps")
        alpha = lr_config.get("alpha")
        warmup_target = lr_config.get("warmup_target")
        warmup_steps = lr_config.get("warmup_steps")

        cosine_decay_schedule = optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            alpha=alpha,
            name = "CosineDecay",
            warmup_target=warmup_target,
            warmup_steps=warmup_steps,
        )
        
        lr_callback = LearningRateScheduler(
            cosine_decay_schedule,
            verbose=1
        )
        callbacks.append(lr_callback)

    elif lr_type == "cyclical":
        # Cyclical Learning Rate
        # TODO: tensorboard shows no actual change to learning rate is happening
        clr_config = config.get("callbacks.cyclical_lr", {})
        base_lr = clr_config.get("base_lr", 0.0003)
        max_lr = clr_config.get("max_lr", 0.001)
        mode = clr_config.get("mode", "triangular2")

        clr = CyclicLR(
            base_lr=base_lr, max_lr=max_lr, step_size=samples_size / batch_size, mode=mode
        )
        callbacks.append(clr)
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
        os.path.join(path_out, "model.h5"),
        monitor="val_loss",
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
        period=1,
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

    if config.get("pruning.prune"): # check if needed with HGQ
        callbacks.extend(
            get_pruning_callbacks(path_out, config.get("pruning.target_sparsity"))
        )

    return callbacks


