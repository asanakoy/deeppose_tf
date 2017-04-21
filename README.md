# DeepPose (stg-1) on TensorFlow

**NOTE**: This is not an official implementation. Original paper is [DeepPose: Human Pose Estimation via Deep Neural Networks](http://arxiv.org/abs/1312.4659).

This is implementation of DeepPose (stg-1).  
Code includes training and testing on 2 popular Pose Benchmarks: [LSP Extended Dataset](http://www.comp.leeds.ac.uk/mat4saj/lspet.html) and [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/).

Performance of Alexnet pretrained on Imagenet and finetuned on LSP is close to the performance reported in the original paper.

### Requirements

- Python 2.7
  - TensorFlow r1.0
  - [Chainer 1.17.0+](https://github.com/pfnet/chainer) (for background data processing only)
  - numpy 1.12+
  - OpenCV 2.4.8+
  - tqdm 4.8.4+
  
For tensorflow version 0.11.0rc0 and 0.12.0rc0 checkout branch [r0.12](https://github.com/asanakoy/deeppose_tf/tree/tensorflow_r0.12)

#### RAM requirements
Requires around 10 Gb of free RAM.

### Installation of dependencies
1. Install TensorFlow
2. Install other dependencies via `pip`.  
```pip install chainer numpy opencv tqdm```
3. In [`scripts/config.py`](scripts/config.py) set `ROOT_DIR` to point to the root dir of the project.
4. Download weights of alexnet pretrained on Imagenet [bvlc_alexnet.tf](https://hcicloud.iwr.uni-heidelberg.de/index.php/s/QBSJhaymOLGxz6M) and put them into [`weights/`](weights/) dir.

### Dataset preparation

```sh
cd datasets
bash download.sh
cd ..
python datasets/lsp_dataset.py
python datasets/mpii_dataset.py
```

- [LSP dataset](http://www.comp.leeds.ac.uk/mat4saj/lsp.html) (1000 tran / 1000 test images)
- [LSP Extended dataset](http://www.comp.leeds.ac.uk/mat4saj/lspet_dataset.zip) (10000 train images)
- **MPII dataset** (use original train set and split it into 17928 train / 1991 test images)
    - [Annotation](http://datasets.d2.mpi-inf.mpg.de/leonid14cvpr/mpii_human_pose_v1_u12_1.tar.gz)
    - [Images](http://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz)

### Training
[examples/](examples/) provide several scripts for training on LSP + LSP_EXT and MPII:
- [examples/train_lsp_alexnet_scratch.sh](examples/train_lsp_alexnet_scratch.sh) to run training Alexnet on LSP + LSP_EXT from scratch
- [examples/train_lsp_alexnet_imagenet.sh](examples/train_lsp_alexnet_imagenet.sh) to run training Alexnet on LSP + LSP_EXT using weights pretrained on Imagenet.
- [examples/train_mpii_alexnet_scratch.py](examples/train_mpii_alexnet_scratch.sh) to run training Alexnet on MPII from scratch.
- [examples/train_mpii_alexnet_imagenet.py](examples/train_mpii_alexnet_imagenet.sh) to run training Alexnet on MPII using weights pretrained on Imagenet.

**Example:** `bash examples/train_lsp_alexnet_scratch.sh`

All these scripts call [`train.py`](scripts/train.py).  
To check which options it accepts and which default values are set, please look into [`cmd_options.py`](scripts/cmd_options.py).

* The network is trained with Adagrad optimizer and learning rate `0.0005` as specified in the paper.  
* For training we use cropped pearsons (not the full image).  
* To use your own network architecure set it accordingly in [`scripts/regressionnet.py`](scripts/regressionnet.py) in `create_regression_net` method.

The network wiil be tested during training and you will see the following output every T iterations:
```
8it [00:06,  1.31it/s]                                                                         
Step 0 test/pose_loss = 0.116
Step	 0	 test/mPCP	 0.005
Step 0 test/parts_PCP:
Head	Torso	U Arm	L Arm	U Leg	L Leg	mean
0.000	0.015	0.001	0.003	0.013	0.001	0.006
Step	 0	 test/mPCKh	 0.029
Step	 0	 test/mSymmetricPCKh	 0.026
Step 0 test/parts_mSymmetricPCKh:
Head	Neck	Shoulder	Elbow	Wrist	Hip	  Knee	Ankle
0.003	0.016	0.019	    0.043	0.044	0.028	0.053	0.003
```
Here you can see that PCP and PCKh scores at step (iteration) 0.  
`test/METRIC_NAME` means that the metric was calculated on test set.  
`val/METRIC_NAME` means that the metric was calculated on validation set. Just for sanity check on LSP I took the first 1000 images from train as validation.  

`pose_loss` is the regression loss of the joint prediction,  
`mPCP` is mean PCP@0.5 score over all sticks,  
`parts_PCP` is PCP@0.5 score for every stick.  
`mRelaxedPCP` is a relaxed PCP@0.5 score, where the stick has a correct position when the average error for both joints is less than the threshold (0.5).   
`mPCKh` is mean PCKh score for all joints,  
`mSymmetricPCKh` is mean PCKh score for all joints, where the score for symmetric left/right joints was averaged,  


### Testing
To test the model use [`tests/test_snapshot.py`](tests/test_snapshot.py).  
- The script will produce PCP@0.5 and PCKh@0.5 scores applied on cropped pearsons.    
- Scores wiil be computed for different crops.   
- BBOX EXTENSION=1 means that the pearson was tightly cropped,    
BBOX EXTENSION=1.5 means that the bounding box of the person was enlarged in 1.5 times and then image was cropped.


**Usage:**  `python tests/test_snapshot.py DATASET_NAME SNAPSHOT_PATH`,   
   - where `DATASET_NAME` is `'lsp'` or `'mpii'`,   
   - `SNAPSHOT_PATH` is the path to the snapshot.   

**Example:** `python tests/test_snapshot.py lsp out/lsp_alexnet_scratch/checkpoint-10000`

### Results
Results for Random initialization and Alexnet initialization from our CVPR 2017 paper [Deep Unsupervised Similarity Learning using Partially Ordered Sets](https://arxiv.org/abs/1704.02268). Check the paper for more results using our initialization and Shuffle&Learn initialization.

#### LSP PCP@0.5

|            | Random Init. | Alexnet |
|------------|--------------|---------|
| Torso      | 87.3         | 92.8    |
| Upper legs | 52.3         | 68.1    |
| Lower legs | 35.4         | 53.0    |
| Upper arms | 25.4         | 39.8    |
| Lower arms | 7.6          | 17.5    |
| Head       | 44.0         | 62.8    |
| Total      | 42.0         | 55.7    |

#### MPII PCKh@0.5
|             | Random Init. | Alexnet |
|-------------|--------------|---------|
| Head        | 79.5         | 87.2    |
| Neck        | 87.1         | 93.2    |
| LR Shoulder | 71.6         | 85.2    |
| LR Elbow    | 52.1         | 69.6    |
| LR Wrist    | 34.6         | 52.0    |
| LR Hip      | 64.1         | 81.3    |
| LR Knee     | 58.3         | 69.7    |
| LR Ankle    | 51.2         | 62.0    |
| Thorax      | 85.5         | 93.4    |
| Pelvis      | 70.1         | 86.6    |
| Total       | 65.4         | 78.0    |

### Notes
If you use this code please cite the repo.  

License
----
GNU General Public License
