pushd src
SET VIDEO_SOURCE=C:\\Users\\yevhe\\PhDProjects\\datasets\\Anti-UAV-RGBT\\test\\20190925_124000_1_5\\visible.mp4
SET MODEL_CONFIG=siamban.toml
start /B python -m demo --debug --video_path %VIDEO_SOURCE% --model_config %MODEL_CONFIG%
popd
pause
