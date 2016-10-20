python ./tools/train_face_detection.py --gpu 0 \
  --net_name VGG16 \
  --weights data/faster_rcnn_models/VGG16_faster_rcnn_final.caffemodel \
  --cfg experiments/cfgs/faster_rcnn_alt_opt.yml
  --imdb voc_2007_trainval
