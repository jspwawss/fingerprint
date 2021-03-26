echo start evalutaing
python3 src/evaluate.py \
--save_weight_only --model=UNet_v8 --checkpoint_path=/home/nmsoc/FPR/Han/fingerprint/checkpoint/kiaraNoise_UNet_v8_2conv_debug/UNet_v8_49-22.94.pb \
--datasetfilename='dataset4perceptual' \
--dataset=kiaraNoise \
--using_CPU \
--debug \
--testing_data=/home/nmsoc/FPR/FVC2000/patch/Db{part}_{mode}/ \

echo test pass^^