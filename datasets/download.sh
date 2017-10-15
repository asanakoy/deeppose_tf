#! /bin/bash
# Script for downloading Datasets: LSP, LSP Extendedd, MPII

# Get LSP Dataset
wget http://sam.johnson.io/research/lsp_dataset_original.zip
unzip lsp_dataset_original.zip
rm -rf lsp_dataset_original.zip
mkdir lsp
mv images lsp/
mv joints.mat lsp/
mv README.txt lsp/

# Get LSP Extended Training Dataset
wget http://sam.johnson.io/research/lspet_dataset.zip
unzip lspet_dataset.zip
rm -rf lspet_dataset.zip
mkdir lsp_ext
mv images lsp_ext/
mv joints.mat lsp_ext/
mv README.txt lsp_ext/

# Get Annotations
wget http://datasets.d2.mpi-inf.mpg.de/leonid14cvpr/mpii_human_pose_v1_u12_1.tar.gz
tar zxvf mpii_human_pose_v1_u12_1.tar.gz
rm -rf mpii_human_pose_v1_u12_1.tar.gz
mv mpii_human_pose_v1_u12_1 mpii

# Get Images
cd mpii
wget http://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz
tar zxvf mpii_human_pose_v1.tar.gz
rm -rf mpii_human_pose_v1.tar.gz

cd ..
