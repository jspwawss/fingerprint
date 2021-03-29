echo start evalutaing
python3 src/evaluate.py \
--save_weight_only --model=UNet_v6 --checkpoint_path=/home/share/Han/novatek/kiaraNoise_UNet_v6_Res95Noskip/UNet_v6_49-0.04.h5 \
--datasetfilename='dataset4perceptual' \
--dataset=kiaraNoise \
--using_CPU \
--testing_data=/home/share/FVC/novatek/dataset/Low/20200923_101219_userN19_left_middle/userN19_left_middle_00_0000.bmp
# \
#--testing_data=/home/nmsoc/FPR/FVC2000/patch/Db{part}_{mode}/ \
#--debug \
echo test pass^^