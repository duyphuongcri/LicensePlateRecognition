
[[Link to my YouTube video!]](https://www.youtube.com/watch?v=oncTZsINEKM)

Step 1: Set up
 python setup.py build
 python setup.py install

Step 2: Download labelling App
[LabelImg GitHub link](https://github.com/tzutalin/labelImg)

Step 3: Training

 python xml_to_csv.py
 python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record
 python generate_tfrecord.py --csv_input=images/test_labels.csv  --image_dir=images/test --output_path=test.record
 python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v2_coco.config

Step 4: Export Inference Graph

 python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph
