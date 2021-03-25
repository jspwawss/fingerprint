echo "start training"
python src/train.py --save_path=/home/nmsoc/FPR/Han/fingerprint/checkpoint/ \
--epoch=50 --model=UNet_v6 \
--model_name=myModel \
--dataset=kiaraNoise \
--dataset_path=/home/nmsoc/FPR/FVC2000/patch/Db{part}_{mode}/ \
--batch_size=1 \
--datasetfilename='dataset4perceptual'  \
\
--losses=mse \
--losses=l1 \
--save_annotation='L1_style_perceptual' \
--losses=perceptual \
--debug




echo "test pass ^^"