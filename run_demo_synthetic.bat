pushd src
SET VIDEO_SOURCE=C:\\Users\\yevhe\\PhDProjects\\datasets\\Synthetic\\train\\UE5.5_Dataset_6.1_Desest\\images
SET GT_PATH=C:\\Users\\yevhe\\PhDProjects\\datasets\\Synthetic\\train\\UE5.5_Dataset_6.1_Desest\\labels.txt
SET MODEL_CONFIG=vit.toml
start /B python -m demo --video_path %VIDEO_SOURCE% --model_config %MODEL_CONFIG% --gt_path %GT_PATH% --data_type synthetic
popd
pause
