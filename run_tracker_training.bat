call "%~dp0find_python.bat"
pushd src
SET CONFIG_PATH=..\\configs\\training.toml
call %PYTHON_CMD% -m models.trainers.train_tracker --config_path %CONFIG_PATH%
popd
