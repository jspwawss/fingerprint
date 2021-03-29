echo "start training"
python src/main.py --save_path=/home/nmsoc/FPR/Han/fingerprint/checkpoint/ \
--epoch=50 --model=UNet_v8 \
--model_name=myModel \
--dataset_path=/home/share/FVC/FVC2000/patch/Db{part}_{mode}/ \
--dataset=kiaraNoise4perceptual \
--batch_size=10 \
--save_annotation='test' \
--losses=mse \
--losses=enMSE \
--losses=perceptual \
--datasetfilename='dataset' \
#--debug \
#


echo "test pass ^^"