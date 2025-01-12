#/bin/bash
# 
# change the following variables to set up code, input and output dirs
#HOST_WORK_DIR="/home/local/PARTNERS/dw640/docker_example"  # change this to your code directory
#HOST_DATA_DIR="/home/local/PARTNERS/dw640/mnt/women_health_internal/DBT"  # change this to your data directory
HOST_WORK_DIR="/home/local/PARTNERS/dw640/Diffusion_for_CT_motion"
HOST_DATA_DIR="/home/local/PARTNERS/dw640/mnt/CAMCA/home/dufan.wu/research_output/ct_motion_diffusion"

# get host user id to populate to the container
HOST_USER_ID="$(id -u)"
HOST_GROUP_ID="$(id -g)"
HOST_USER_NAME=${USER}
CONTAINER_WORKDIR="/workspace/Diffusion_for_CT_motion" 
CONTAINER_DATADIR="/data" #should be /data when using NAS drive   

sudo docker run -it --rm --gpus=all --name=ct_motion_diffusion --network="bridge" --shm-size=256m --ipc=host \
-v ${HOST_WORK_DIR}:${CONTAINER_WORKDIR} \
-v ${HOST_DATA_DIR}:${CONTAINER_DATADIR} \
-e CONTAINER_UID=${HOST_USER_ID} \
-e CONTAINER_GID=${HOST_GROUP_ID} \
-e CONTAINER_UNAME=${HOST_USER_NAME} \
-e CONTAINER_WORKDIR=${CONTAINER_WORKDIR} \
ct_motion_diffusion/inference:1.0