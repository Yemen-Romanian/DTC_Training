pushd src
SET VIDEO_SOURCE=C:\\Users\\yevhe\\PhDProjects\\datasets\\Synthetic\\train\\UE5.5_Dataset_6.1_Desest\\images
SET GT_PATH=C:\\Users\\yevhe\\PhDProjects\\datasets\\Synthetic\\train\\UE5.5_Dataset_6.1_Desest\\labels.txt
start /B python -m demo --video_path %VIDEO_SOURCE% --gt_path %GT_PATH% --data_type synthetic
popd
pause
