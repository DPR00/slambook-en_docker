# Slambook-en using Docker

This repository contains the code from [slambook-en](https://github.com/gaoxiang12/slambook-en) to run it with Docker. The slambook-en is the English version of 14 lectures on visual SLAM. Currently, there are 6 chapters in the `code` folder because I'm in that chapter :laughing:. I will be updating the folder as soon as I continue reading the book (but all the code are in the repo of the book).

**Prerequisite:** You have to [install docker](https://docs.docker.com/desktop/install/ubuntu/) in your computer.

*Disclaimer:* This repo has been tested in Ubuntu 20.04 and WSL 2 (Ubuntu 22.04).

## Build the image

To build the image, set your terminal in this folder and the run the `build.sh` bash script:

```
./build.sh
```

You can omit `sudo` in case you set the Docker daemon to run as a non-root user ([rootless mode](https://docs.docker.com/engine/security/rootless/)).

Also, I am using this repository in WSL, and you could face some errors such as: `WARNING: buildx: failed to read current commit information with git rev-parse --is-inside-work-tree Dockerfile:9`. I found that this error occurs because docker tries to run `docker buildx build` instead of `docker build`. So, if you want to use this folder in WSL2, run the following bash script:
```
./buildwsl.sh
```

## Create a container

To create a container, run the `runCnt.sh` bash script:

```
./runCnt.sh
```

## Interact with the container

You can work in the container you've just created.
I prefer to work in vs-code as root (not recommended, but I like it). If you hve docker in rootles mode, then it is not necessary to run vscode as root.

If you like to try (At your own risk), run vscode as root:
```
sudo code --user-data-dir="~/.vscode-root"
```
Then, install the following extensions:
- Docker (ms-azuretools.vscode-docker)
- Remote Explorer (ms-vscode.remote-explorer)
- Dev Containers (ms-vscode-remote.remote-)

Now, you can start, stop or manage any container from vs-code (Docker extension). After start the container, you can open a new vs-code window from Remote Explorer -> Dev Container. You should see all the started containers there.

Once you have the vs-code window for your container, then you can setup your own environment to work with the C++ files of the book in vs-code. The files in the `config` folder are my config files for vs-code.

