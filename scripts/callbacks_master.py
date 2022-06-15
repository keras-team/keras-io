"""
deliberately skipped: BaseLogger, History
kept: ProgbarLogger
"""

CALLBACKS_MASTER = {
    "path": "callbacks/",
    "title": "Callbacks API",  # TODO
    "toc": True,
    "children": [
        {
            "path": "base_callback",
            "title": "Base Callback class",
            "generate": ["tensorflow.keras.callbacks.Callback"],
        },
        {
            "path": "model_checkpoint",
            "title": "ModelCheckpoint",
            "generate": ["tensorflow.keras.callbacks.ModelCheckpoint"],
        },
        {
            "path": "backup_and_restore",
            "title": "BackupAndRestore",
            "generate": ["tensorflow.keras.callbacks.BackupAndRestore"],
        },
        {
            "path": "tensorboard",
            "title": "TensorBoard",
            "generate": ["tensorflow.keras.callbacks.TensorBoard"],
        },
        {
            "path": "early_stopping",
            "title": "EarlyStopping",
            "generate": ["tensorflow.keras.callbacks.EarlyStopping"],
        },
        {  # LEGACY
            "path": "learning_rate_scheduler",
            "title": "LearningRateScheduler",
            "generate": ["tensorflow.keras.callbacks.LearningRateScheduler"],
        },
        {
            "path": "reduce_lr_on_plateau",
            "title": "ReduceLROnPlateau",
            "generate": ["tensorflow.keras.callbacks.ReduceLROnPlateau"],
        },
        {
            "path": "remote_monitor",
            "title": "RemoteMonitor",
            "generate": ["tensorflow.keras.callbacks.RemoteMonitor"],
        },
        {
            "path": "lambda_callback",
            "title": "LambdaCallback",
            "generate": ["tensorflow.keras.callbacks.LambdaCallback"],
        },
        {
            "path": "terminate_on_nan",
            "title": "TerminateOnNaN",
            "generate": ["tensorflow.keras.callbacks.TerminateOnNaN"],
        },
        {
            "path": "csv_logger",
            "title": "CSVLogger",
            "generate": ["tensorflow.keras.callbacks.CSVLogger"],
        },
        {
            "path": "progbar_logger",
            "title": "ProgbarLogger",
            "generate": ["tensorflow.keras.callbacks.ProgbarLogger"],
        },
    ],
}
