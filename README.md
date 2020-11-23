#### Prepare and process data
##### VRD dataset
`mkdir data && cd data
wget http://cs.stanford.edu/people/ranjaykrishna/vrd/json_dataset.zip
wget http://imagenet.stanford.edu/internal/jcjohns/scene_graphs/sg_dataset.zip
unzip json_dataset.zip -d vrd
unzip json_dataset.zip -d vrd/images
mv vrd/images/sg_train_images/\* vrd/images/
mv vrd/images/sg_test_images/\* vrd/images/
rm json_dataset.zip
cd ..
python process_data vrd
`

##### VG200 dataset
Download annotation data following the instruction in https://github.com/zawlin/cvpr17_vtranse
create dataset folder : `mkdir vg200`
move `vg1_2_meta.h5` under `vg200`
Download image data fom https://visualgenome.org/, and extract images to `data/vg200/images`
process data : `python process_data vg200`

#### Run
##### Train model
`
cd ./TCN_VRP_pytorch/
python process_data.py vrd
`

