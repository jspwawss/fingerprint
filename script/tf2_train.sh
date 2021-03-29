echo "start training"
python src/tf2_train.py --save_path=/home/share/Han/novatek/ \
--epoch=100 --model=UNet_v8 \
--batch_size=10 \
--model_name=myModel \
--dataset=kiaraNoise \
--dataset_path=/home/share/FVC/FVC2000/patch/Db{part}_{mode}/ \
--batch_size=1 \
--datasetfilename='dataset4perceptual'  \
--print_rate=1000 \
--save_annotation='107' \
--losses=mse \
--lr=0.0001 \
--beta_1=0.85 \
--beta_2=0.995 \
#--debug
#--losses=enMSE \
#--losses=perceptual \
#--losses=l1 \
#
#
#
#


echo "test pass ^^"