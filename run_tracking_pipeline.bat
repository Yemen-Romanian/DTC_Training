pushd src
:: SET VIDEO_SOURCE=C:\\Users\\yevhe\\PhDProjects\\datasets\\UAV123\\data_seq\\UAV123\\bike3
SET VIDEO_SOURCE=C:\\Users\\yevhe\\PhDProjects\\datasets\\vot_2022_st\\tiger\\color
SET OUT=output
SET MODEL_PATH=""
start /B python -m pipelines.tracking_pipeline --model_path %MODEL_PATH% --image_folder_path %VIDEO_SOURCE% --output_path %OUT%
popd
pause
