## Train
    python train.py --config data_configs/fishes.yaml --epochs 70 --batch-size 10 --results-folder fishes --use-train-aug --no-mosaic

### Training Plots
![mAP](images/train/mAP.png)
![train_loss_epoch](images/train/train_loss_epoch.png)
![train_loss_iter](images/train/train_loss_iter.png)
![Box regression loss](images/train/loss_bbox_reg.png)
![Classification loss](images/train/loss_cls.png)
![Object loss](images/train/loss_obj.png)
![RPN bounding box loss](images/train/loss_rpn_bbox.png)

## Inference
    ​python inference.py --weights runs/train/fishes/best_model.pth --input D:/CV/datasets/fishes_voc/test

![mAP](images/inference/clownfish.jpg)
![mAP](images/inference/OrchidDottyback.jpg)
![mAP](images/inference/rainbowfish.jpg)
![mAP](images/inference/yellowtang.jpg)
![mAP](images/inference/zebrafish.jpg)