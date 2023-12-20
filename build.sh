# !/bin/bash

SCRIPT_FOLDER_PATH="$(cd "$(dirname "$0")"; pwd)"

IMAGE_NAME="slambook_en"
DOCKERFILE_PATH=$SCRIPT_FOLDER_PATH/Dockerfile

USERID=$(id -u)
USER=$(whoami)

# echo "SCRIPT_FOLDER_PATH: ${SCRIPT_FOLDER_PATH}"
# echo "CONTEXT_FOLDER_PATH: ${CONTEXT_FOLDER_PATH}"
# echo "IMAGE_NAME: ${IMAGE_NAME}"
# echo "DOCKERFILE_PATH: ${DOCKERFILE_PATH}"
# echo "USERID: ${USERID}"
# echo "USER: ${USER}"

sudo docker build -t $IMAGE_NAME \
     --file $DOCKERFILE_PATH \
     --build-arg USERID=$USERID \
     --build-arg USER=$USER \
     $SCRIPT_FOLDER_PATH