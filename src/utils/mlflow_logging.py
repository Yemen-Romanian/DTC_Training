"""DTC-specific glue between the training pipeline and the generic MLFlower wrapper.

Keeps MLflow-domain translation (Config -> params, raw metric dicts -> namespaced
metrics, dataset paths -> MLflow dataset descriptors) out of both the Trainer and the
reusable MLFlower class.
"""
from datasets.dataset_factory import create_dataset


def _build_dataset_descriptors(config) -> list:
    """Build MLflow dataset descriptors for each split/source from the config paths.

    Re-parses the datasets to collect per-sequence labels, so the logged lineage lists
    exactly which sequences were used in train/val/test.
    """
    splits = {
        'training': config.get_train_paths(),
        'validation': config.get_val_paths(),
        'testing': config.get_test_paths() or {},
    }
    descriptors = []
    for context, paths_dict in splits.items():
        for source_name, path in paths_dict.items():
            videos = create_dataset(source_name, path).parse()
            descriptors.append({
                'name': f"{context}_{source_name}",
                'source': path,
                'context': context,
                'samples': [video.label for video in videos],
            })
    return descriptors


def save_training_run(mlflower, config, model, val_metrics: dict, test_metrics: dict,
                      best_model_path=None, registered_model_name=None):
    """Push a completed training run to MLflow.

    Translates DTC-domain objects into a generic ``MLFlower.save_experiment`` call:
    - params are read from the training/model config,
    - raw val/test metric dicts are namespaced with ``val_``/``test_`` prefixes,
    - dataset descriptors are derived from the config's split paths,
    - the model is logged on CPU so it reloads without a GPU.

    Returns the MLflow run id.
    """
    model_config = config.get_model_config()
    experiment_name = config.get_param("experiment_name") or f"{model_config['id']}_training"

    params = {
        'model_id': model_config['id'],
        'backbone': model_config['backbone']['type'],
        'batch_size': config.get_training_param('batch_size'),
        'lr': config.get_training_param('lr'),
        'epochs_num': config.get_training_param('epochs_num'),
    }

    metrics = {f"val_{name}": value for name, value in val_metrics.items()}
    metrics.update({f"test_{name}": value for name, value in test_metrics.items()})

    artifacts = [str(best_model_path)] if best_model_path else []

    # Log on CPU so the model reloads on machines without a GPU (no map_location needed).
    model.to('cpu')
    dataset_descriptors = _build_dataset_descriptors(config)
    params['train_size'] = sum(len(d['samples']) for d in dataset_descriptors if d['context'] == 'training')
    params['validation_size'] = sum(len(d['samples']) for d in dataset_descriptors if d['context'] == 'validation')
    params['test_size'] = sum(len(d['samples']) for d in dataset_descriptors if d['context'] == 'testing')

    return mlflower.save_experiment(
        experiment_name=experiment_name,
        params=params,
        metrics=metrics,
        model=model,
        registered_model_name=registered_model_name,
        datasets=dataset_descriptors,
        artifacts=artifacts,
    )
