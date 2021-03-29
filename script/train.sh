echo "start training"
python src/train.py --save_path=/home/share/Han/novatek/ \
--epoch=50 --model=UNet_v8 \
--model_name=myModel \
--dataset=kiaraNoise \
--dataset_path=/home/share/FVC/FVC2000/patch/Db{part}_{mode}/ \
--batch_size=1 \
--datasetfilename='dataset4perceptual'  \
--print_rate=1000 \
--save_annotation='test' \
--losses=mse \
--losses=enMSE \
--losses=perceptual \
#--debug
#--losses=l1 \
#
#


echo "test pass ^^"