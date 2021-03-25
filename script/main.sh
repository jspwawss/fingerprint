echo "start training"
python src/main.py --save_path=/home/nmsoc/FPR/Han/fingerprint/checkpoint/ \
--epoch=50 --model=UNet_v6 \
--model_name=myModel \
--dataset_path=/home/nmsoc/FPR/FVC2000/noise_patch/Db{part}_{mode}/ \
--dataset=kiaraNoise \
--batch_size=10 \
--save_annotation='v1' \
--debug \
#--datasetfilename='dataset4perceptual' \


echo "test pass ^^"