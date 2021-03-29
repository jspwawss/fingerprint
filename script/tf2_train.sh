echo "start training"
python src/tf2_train.py --save_path=/home/share/Han/novatek/ \
--epoch=5 --model=UNet_v8 \
--batch_size=10 \
--model_name=myModel \
--dataset=kiaraNoise \
--dataset_path=/home/share/FVC/FVC2000/patch/Db{part}_{mode}/ \
--batch_size=1 \
--datasetfilename='dataset4perceptual'  \
--print_rate=1000 \
--save_annotation='test' \
--losses=mse \
#--losses=enMSE \
#--losses=perceptual \
#--losses=l1 \
#--debug
#
#
#


echo "test pass ^^"