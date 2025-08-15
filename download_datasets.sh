# don't directly execute this script
# manually copy and execute below lines to download each dataset separatelly

mkdir -p isegm_datasets
cd isegm_datasets

# Grab Cut done
wget https://github.com/saic-vul/fbrs_interactive_segmentation/releases/download/v1.0/GrabCut.zip
unzip GrabCut.zip

# Barkeley done
wget https://github.com/saic-vul/fbrs_interactive_segmentation/releases/download/v1.0/Berkeley.zip
unzip Berkeley.zip

# DAVIS done
wget https://github.com/saic-vul/fbrs_interactive_segmentation/releases/download/v1.0/DAVIS.zip
unzip DAVIS.zip

# COCO_MVal done
wget https://github.com/saic-vul/fbrs_interactive_segmentation/releases/download/v1.0/COCO_MVal.zip
unzip COCO_MVal.zip

# SBD done
wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz -O SBD.tgz
mkdir -p SBD && tar -xzvf SBD.tgz -C SBD --strip-components 1

# Pascal VOC 2012 done
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_11-May-2012.tar --strip-components 1

# MS COCO
wget http://images.cocodataset.org/zips/train2017.zip -O MSCOCO_train.zip
wget http://images.cocodataset.org/zips/val2017.zip -O MSCOCO_val.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O MSCOCO_annotations.zip
wget http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip -O MSCOCO_panoptic_annotations.zip
mkdir -p MSCOCO 
unzip MSCOCO_train.zip -d MSCOCO
unzip MSCOCO_val.zip -d MSCOCO
unzip MSCOCO_annotations.zip -d MSCOCO
unzip MSCOCO_panoptic_annotations.zip -d MSCOCO
unzip -j MSCOCO/annotations/panoptic_train2017.zip -d MSCOCO/annotations/panoptic_train
unzip -j MSCOCO/annotations/panoptic_val2017.zip -d MSCOCO/annotations/panoptic_val
mv MSCOCO/annotations/panoptic_train2017.json MSCOCO/annotations/panoptic_train.json  
mv MSCOCO/annotations/panoptic_val2017.json MSCOCO/annotations/panoptic_val.json  
mv MSCOCO/val2017 MSCOCO/val && mv MSCOCO/train2017 MSCOCO/train

# LVIS v1.0 (use MS COCO images, labels only)
wget https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip
wget https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip
mkdir -p LVIS/train
mkdir -p LVIS/val
unzip lvis_v1_train.json.zip -d LVIS
unzip lvis_v1_val.json.zip -d LVIS
mv LVIS/lvis_v1_train.json LVIS/train/lvis_train.json
mv LVIS/lvis_v1_val.json LVIS/val/lvis_val.json
# soft link to connect MS COCO images
ln -s $PWD/MSCOCO/train $PWD/LVIS/train/images
ln -s $PWD/MSCOCO/val $PWD/LVIS/val/images

# COCO + LVIS (use MS COCO images, labels only) done
wget https://github.com/saic-vul/ritm_interactive_segmentation/releases/download/v1.0/cocolvis_annotation.tar.gz
tar -xvf cocolvis_annotation.tar.gz -C LVIS/
