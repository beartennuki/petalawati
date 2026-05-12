from prefect import task
from prefect.cache_policies import NO_CACHE
from app.job_store import append_metric, model_path, confusion_path
import json


@task(name="train-model", cache_policy=NO_CACHE)
def train_model(model, train_ds, val_ds, job_id: str, epochs: int, learning_rate: float):
    import tensorflow as tf

    class EpochLogger(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            append_metric(job_id, {
                "epoch": epoch + 1,
                "loss": round(float(logs.get("loss", 0)), 4),
                "val_loss": round(float(logs.get("val_loss", 0)), 4),
                "accuracy": round(float(logs.get("accuracy", 0)), 4),
                "val_accuracy": round(float(logs.get("val_accuracy", 0)), 4),
            })

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[EpochLogger()],
        verbose=0,
    )
    return model


@task(name="evaluate-model", cache_policy=NO_CACHE)
def evaluate_model(model, val_ds, class_names: list[str], job_id: str):
    import numpy as np
    from sklearn.metrics import confusion_matrix

    y_true, y_pred = [], []
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().argmax(axis=1))
        y_pred.extend(preds.argmax(axis=1))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    confusion_path(job_id).write_text(json.dumps(cm.tolist()))


@task(name="save-model", cache_policy=NO_CACHE)
def save_model(model, job_id: str):
    model.save(str(model_path(job_id)))
