# Uses uv (project .venv) if installed, otherwise python3 on PATH.
if command -v uv >/dev/null 2>&1; then
    PYTHON_CMD="uv run python"
else
    PYTHON_CMD="python3"
fi
echo "Using interpreter: $PYTHON_CMD"

cd src
CONFIG_PATH=../configs/training.toml
$PYTHON_CMD -m models.trainers.train_tracker --config_path $CONFIG_PATH
