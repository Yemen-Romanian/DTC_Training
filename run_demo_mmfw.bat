pushd src
SET SEQUENCE=58
SET DATASET_ROOT=C:\\Users\\yevhe\\PhDProjects\\datasets\\MMFW-UAV-Processed\\Wide-Angle-Sensor-Subset
SET VIDEO_SOURCE=%DATASET_ROOT%\\Sequences\\%SEQUENCE%
SET GT_PATH=%DATASET_ROOT%\\Annotations_new\\%SEQUENCE%.txt
SET MODEL_CONFIG=siamban.toml
start /B python -m demo --video_path %VIDEO_SOURCE% --model_config %MODEL_CONFIG% --gt_path %GT_PATH% --data_type mmfw
popd
pause
