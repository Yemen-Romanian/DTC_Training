call "%~dp0find_python.bat"
pushd src
SET EVAL_CONFIG=evaluation.toml
SET MODEL_CONFIG=siamban.toml
call %PYTHON_CMD% -m evaluation.tracker_evaluation --evaluation_config %EVAL_CONFIG% --model_config %MODEL_CONFIG%
popd
pause
