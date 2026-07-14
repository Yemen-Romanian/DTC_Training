pushd src
SET EVAL_CONFIG=evaluation.toml
SET MODEL_CONFIG=vit.toml
call python -m evaluation.tracker_evaluation --evaluation_config %EVAL_CONFIG% --model_config %MODEL_CONFIG%
popd
pause
