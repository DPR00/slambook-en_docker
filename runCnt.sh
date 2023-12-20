xhost +local:docker
XSOCK=/tmp/.X11-unix
IMAGE_NAME="slambook_en"
# XAUTH=/tmp/.docker.xauth
# chmod 755 $XAUTH
# xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -
sudo docker run -it \
        --name="cnt_slambook" \
        -e DISPLAY=$DISPLAY \
        -v $XSOCK:$XSOCK \
        -v ./code:/slambook-en_notes \
        slambook_en bash
