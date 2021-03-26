echo "start training"
python src/train.py --save_path=/home/nmsoc/FPR/Han/fingerprint/checkpoint/ \
--epoch=50 --model=UNet_v7 \
--model_name=myModel \
--dataset=kiaraNoise \
--dataset_path=/home/nmsoc/FPR/FVC2000/patch/Db{part}_{mode}/ \
--batch_size=1 \
--datasetfilename='dataset4perceptual'  \
--print_rate=1000 \
--save_annotation='skip' \
--losses=mse \
--losses=l1 \
--losses=enMSE \
#--losses=perceptual \
#--debug


echo "start training"
python src/train.py --save_path=/home/nmsoc/FPR/Han/fingerprint/checkpoint/ \
--epoch=50 --model=UNet_v8 \
--model_name=myModel \
--dataset=kiaraNoise \
--dataset_path=/home/nmsoc/FPR/FVC2000/patch/Db{part}_{mode}/ \
--batch_size=1 \
--datasetfilename='dataset4perceptual'  \
--print_rate=1000 \
--save_annotation='edgeDetect' \
--losses=mse \
--losses=enMSE \
--losses=l1 \
#--losses=perceptual \
#--debug


echo "test pass ^^"