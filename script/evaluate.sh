echo start evalutaing
python3 src/evaluate.py \
--save_weight_only --model=UNet_v6 --checkpoint_path=/home/nmsoc/FPR/Han/fingerprint/checkpoint/kiaraNoise_UNet_v6_L1_style_perceptual_debug/UNet_v6_49-41.29.h5 \
--input_shape=50 \
--datasetfilename='dataset4perceptual' \
--dataset=kiaraNoise \
--using_CPU \
--debug \
--testing_data=/home/nmsoc/FPR/FVC2000/patch/Db{part}_{mode}/ \

echo test pass^^