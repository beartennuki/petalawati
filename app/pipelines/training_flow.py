from prefect import flow, get_run_logger
from app.job_store import get_job, update_job
from app.pipelines.tasks.data_tasks import extract_dataset, build_datasets
from app.pipelines.tasks.model_tasks import build_model
from app.pipelines.tasks.train_tasks import train_model, evaluate_model, save_model
from app.config import UPLOADS_DIR


@flow(name="training-flow", log_prints=True)
def training_flow(job_id: str):
    logger = get_run_logger()
    logger.info(f"Starting training flow for job {job_id}")

    update_job(job_id, status="running")

    try:
        job = get_job(job_id)
        cfg = job.config

        # Data
        zip_path = str(UPLOADS_DIR / f"{job_id}.zip")
        dataset_info = extract_dataset(zip_path, job_id)
        train_ds, val_ds, class_names = build_datasets(
            data_dir=dataset_info["data_dir"],
            image_size=cfg.image_size,
            batch_size=cfg.batch_size,
            val_split=cfg.val_split,
            augment=cfg.augment,
        )

        # Model
        model = build_model(
            architecture=cfg.architecture,
            num_classes=len(class_names),
            image_size=cfg.image_size,
            freeze_base=cfg.freeze_base,
        )

        # Train
        trained_model = train_model(
            model=model,
            train_ds=train_ds,
            val_ds=val_ds,
            job_id=job_id,
            epochs=cfg.epochs,
            learning_rate=cfg.learning_rate,
        )

        # Evaluate + save
        evaluate_model(trained_model, val_ds, class_names, job_id)
        save_model(trained_model, job_id)

        update_job(job_id, status="completed")
        logger.info(f"Job {job_id} completed successfully")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        update_job(job_id, status="failed", error=str(e))
        raise
