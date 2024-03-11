#!/bin/bash
APP_IMG_NAME="magmaforge-img"
BASE_DIR="/home/jovyan"
if [ -z "$1" ]
then 
	echo "Accessing a Jupyter Lab ThermoEngine container rooted to this directory ..."
	docker build -t $APP_IMG_NAME .
	docker run -p 8888:8888 \
               --env JUPYTER_ENABLE_LAB=yes \
               --user root \
               -e GRANT_SUDO=yes \
               -v $PWD:$BASE_DIR/app \
               $APP_IMG_NAME start-notebook.sh
 elif [ "$1" = "term" ]
 then
 	echo "Running an interactive shell into a ThermoEngine container ..."
	docker build -t $APP_IMG_NAME .
    docker run -it \
               -v $PWD:$BASE_DIR/app \
               --rm $APP_IMG_NAME bash
 elif [ "$1" = "stop" ]
 then
 	echo "Removing running docker containers ..."
 	if [ ! -z "$(docker ps -q)" ]
 	then
 		docker kill $(docker ps -q)
 	fi
 else
 	echo "Usage:"
 	echo "  <no argument> - Run latest ThermoEngine docker container from GitLab."
 	echo "  term - Run a terminal shell in the latest ThermoEngine docker container from GitLab."
 	echo "  stop - Remove any running docker containers from your system."
 	echo "  help - This message."
 fi