# DeepPose (stg-1) on TensorFlow

**NOTE**: This is not an official implementation. Original paper is [DeepPose: Human Pose Estimation via Deep Neural Networks](http://arxiv.org/abs/1312.4659).

This is implementation of DeepPose (stg-1).
Code includes training and testing on 2 popular Pose Benchmarks: [LSP Extended Dataset](http://www.comp.leeds.ac.uk/mat4saj/lspet.html) and [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/).

### Requirements

- Python 2.7
  - [TensorFlow 0.11.0rc0](https://github.com/tensorflow/tensorflow/releases/tag/v0.11.0rc0) (I didn't test on later versions)
  - [Chainer 1.17.0+](https://github.com/pfnet/chainer) (for background data processing only)
  - numpy 1.12+
  - OpenCV 2.4.8+
  - tqdm 4.8.4+

#### RAM requirement
Requires around 10 Gb of free RAM.

### Installation of dependencies
1. Install TensorFlow
2. Install other dependencies via `pip`.
    ```sh
    pip install chainer
    pip install numpy
    pip install opencv
    ```
3. In [scripts/config.py](scripts/config.py) set `ROOT_DIR` to the point to the root dir of the project.
4. Download weights of alexnet pretrained on Imagenet [bvlc_alexnet.tf](bvlc_alexnet.tf) and move them into [`weights/`](weights/) dir.

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
- **MPII dataset** (use validation set and split in into 17928 train / 1991 test images)
    - [Annotation](http://datasets.d2.mpi-inf.mpg.de/leonid14cvpr/mpii_human_pose_v1_u12_1.tar.gz)
    - [Images](http://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz)

### Training
[examples/](examples/) provide several scripts for training on LSP + LSP_EXT and MPII:
- [examples/train_lsp_alexnet_scratch.sh](examples/train_lsp_alexnet_scratch.sh) to run training Alexnet on LSP + LSP_EXT from scratch
- [examples/train_lsp_alexnet_imagenet.sh](examples/train_lsp_alexnet_imagenet.sh) to run training Alexnet on LSP + LSP_EXT using weights pretrained on Imagenet.
- [examples/train_mpii_alexnet_scratch.py](examples/train_mpii_alexnet_scratch.sh) to run training Alexnet on MPII from scratch.
- [examples/train_mpii_alexnet_imagenet.py](examples/train_mpii_alexnet_imagenet.sh) to run training Alexnet on MPII using weights pretrained on Imagenet.

All these scripts call [`train.py`](scripts/train.py).
To check which options it accepts and which default values are set, please look into [`cmd_options.py`](scripts/cmd_options.py).

The network is trained with Adagrad optimizer and learning rate `0.0005` as specified in the paper.
For training we used cropped pearsons (not the full image).
To use your own network architecure set it accordingly in [`scripts/regressionnet.py`](scripts/regressionnet.py) in `create_regression_net` method.

### Testing
To test the model use [`tests/test_snapshot.p`](tests/test_snapshot.py).
The script will produce PCP and PCKh scores applied on cropped pearsons.
Scores wiil be computed for different crops.
BBOX EXTENSION=1 means that the pearson was tightly cropped,
BBOX EXTENSION=1.5 means that the bounding box of the person was enlarged in 1.5 times and then image was cropped.


**Usage:**  `python tests/test_snapshot.py DATASET_NAME SNAPSHOT_PATH`,
   where `DATASET_NAME` is `'lsp'` or `'mpii'`,
   `SNAPSHOT_PATH` is the path to the snapshot.
**Example:** `python tests/test_snapshot.py lsp out/lsp_alexnet_scratch/checkpoint-10000`



License
----
GNU General Public License
