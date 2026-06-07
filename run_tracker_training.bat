pushd src
SET CONFIG_PATH=..\\configs\\training.toml
call python -m models.trainers.train_tracker --config_path %CONFIG_PATH%
popd
