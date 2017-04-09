import sys
import shlex
import os.path
import scripts.train
import scripts.config

argv = """
--max_iter 1000000 \
--batch_size 128 \
--snapshot_step 5000 \
--test_step 250 \
--log_step 2 \
--dataset_name mpii
--train_csv_fn {0}/datasets/mpii/train_joints.csv \
--test_csv_fn {0}/datasets/mpii/test_joints.csv \
--val_csv_fn '' \
--img_path_prefix {0}/datasets/mpii/images \
--should_downscale_images \
--downscale_height 400 \
--n_joints 16 \
--seed 1701 \
--im_size 227 \
--min_dim 6 \
--shift 0.1 \
--bbox_extension_min 1.0 \
--bbox_extension_max 1.2 \
--coord_normalize \
--fname_index 0 \
--joint_index 1 \
--ignore_label -100500 \
--symmetric_joints "[[12, 13], [11, 14], [10, 15], [2, 3], [1, 4], [0, 5]]" \
--conv_lr 0.0005 \
--fc_lr 0.0005 \
--fix_conv_iter 0 \
--optimizer adagrad \
--o_dir {0}/out/mpii_alexnet_scratch \
--gcn \
--fliplr \
--workers 4 \
--net_type Alexnet \
--reset_iter_counter
""".format(scripts.config.ROOT_DIR)

argv = shlex.split(argv)
print argv
scripts.train.main(argv)
