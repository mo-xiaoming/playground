## installation on ubuntu20.04
```
sudo apt update

sudo apt install apt-transport-https ca-certificates curl software-properties-common

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu focal stable"

sudo apt update

apt-cache policy docker-ce

sudo apt install docker-ce

sudo systemctl status docker

sudo usermod -aG docker $USER

newgrp
```

## list all docker images in a registry

`http://localhost:5000/v2/_catalog`

`http://localhost:5000/v2/<repo>/<image>/tags/list`

on client

add to /etc/docker/daemon.json

`{"insecure-registries":["192.168.0.13:5000"]}`

then

`sudo systemctl restart docker`

## basic commands

###image history
`docker history <image>`

###run an docker, pull if necessary
`docker run hello-world`

###Create an container
`docker create -it ubuntu:20.04 bash`

###list all containers
`docker ps -a`

###start a container
`docker start <container id>`

###combine `create` and `start`, detached
`docker run -it -d ubuntu:20.04 bash`

###automatically remove container when exit, --rm
`docker run -it --rm -d ubuntu:20.04 bash`

###attach to a running container
`docker attach <container id>`

###attach to a running container with a new command
`docker exec -it <container id> <cmd>`

###stop docker
`docker stop <container id>`

###detach from inside a container
`Ctrl-p Ctrl-q`

###check logs
`docker logs <container id>`

###insepct a docker
`docker inspect <container id>`

###environment vairable
`docker run -e KEY=VALUE <image>`

###peristent storage
`docker create -it -v $(pwd):/var/www ubuntu:latest bash`

###mapping ports HOST_PORT:VM_PORT
`docker run --name webserver -v $(pwd):/usr/share/nginx/html -d -p 8080:80 nginx`

## command in details

### `CMD` vs `ENTRYPOINT`

`CMD sleep 5`, that's it, unless `docker run <container id> sleep 10`

`ENTRYPOINT sleep`, pass in parameters via `docker run <container id> 10`

To give a default value for `ENTRYPOINT`

```dockerfile
ENTRYPOINT sleep
CMD 5
```

`5` is the default value for `sleep` if it is not specified in command line

To override `ENTRYPOINT`, `docker run --entrypoint sleep2.0 <container id> 10`, command becomes `sleep2.0 10`

### networks

- Bridge: internal address `172.17.0.x`, has to be mapped to outside world
- none: `--network=none`
- host: `--network=host`, without mapping, but cannot be mapped to the same port

#### create new network

`docker network create --driver bridge --subnet 182.18.0.0/16 <some network name>`

### volumes

- without `/` is volume mounting, which inside `/var/lib/docker`

  `docker volume create date_volume` creates a `data_volume` folder under `/var/lib/docker`

  `docker run -v data_volume:/var/lib/mysql mysql` maps this folder to container's `/var/lib/mysql`

  Or

  `docker run -v data_volume2:/var/lib/mysql mysql` directly create new volume

new syntax

`docker run --mount type=bind,source=/data/mysql,target=/var/lib/mysq mysql`

- with absolute path is bind mounting, which can be anywhere

### connect containers

`docker run -d --name=vote -p 5000:80 --link redis:redis voting-app`

`--link <container name>:<hostname in /etc/hosts>` add an entry `<ip>    <hostname>` in `/etc/host` of `vote`

which translates to `docker-compose.yml`

```yaml
version: 2
services:
  redis:
    image: redis
  vote:
    build: ./vote # optionally specify where Dockerfile is
    image: voting-app
    ports:
    - 5000:80
    depends_on:
    - redis
```

```yaml
version: 2
services:
  redis:
    image: redis
	networks:
	- back-end
  db:
    image: postgres:9.4
	networks:
	- back-end
  vote:
    image: voting-app
	networks:
	- front-end
	- back-end
  result:
    image: result
	networks:
	- front-end
	- back-end

networks:
  front-end:
  back-end:
```

## exmaples

###create my own image
```dockerfile
FROM ubuntu

RUN apt-get update && apt-get install -y python

RUN pip install flask flask-mysql

COPY . /opt/source-code

ENTRYPOINT FLASK_APP=/opt/source-code/app.py flask run
```

###create custorized image
```dockerfile
FROM nginx:alpine
VOLUME /usr/share/nginx/html
EXPOSE 80
```

```bash
docker build . -t webserver:v1
docker run --name webserver -v $(pwd):/usr/share/nginx/html -d -p 8080:80 webserver:v1
```

###multiple build
```dockerfile
FROM adoptopenjdk:11-jre-hotspot as builder
WORKDIR application
ARG JAR_FILE=target/*.jar
COPY ${JAR_FILE} application.jar
RUN java -Djarmode=larytools -jar application.jar extract

FROM adoptopen:11-jre-hotspot
WORKDIR application
COPY --from=builder application/dependencies/ ./
COPY --from=builder application/snapshot-dependencies/ ./
COPY --from=builder application/resources/ ./
COPY --from=builder application/application/ ./
ENTRYPOINT ["java", "org.springframework.boot.loader.JarLancher"]
```

###list images
`docker images`

###delete image
`docker rmi <image id>`

##Build

##Share
###Pull an image from a registry
```docker pull myimage:1.0```

###Retag a local image with a new image name and tag
```docker tag myimage:1.0 myrepo/myimage:2.0```

###Push an image to a registry
```docker push myrepo/myimage:2.0```

##Run

###Run a container from the Alpine version 3.9 image, name the running container “web” and expose port 5000 externally, mapped to port 80 inside the container.

```docker run --name web -p 5000:80 alpine:3.9```

###Stop a running container through SIGTERM
```docker container stop web```

###Stop a running container through SIGKILL
```docker container kill web```

###create a new image from a container
`docker commit <container> <repo>/<image>`

###docker container -> file
`docker export --output="<file>" <container>`

###docker image -> file
`docker save <image> > /tmp/file.tar`

###file -> docker image
`docker import <file>`

###List the networks
```docker network ls```

###Delete all running and stopped containers
```docker container rm -f $(docker ps -aq)```

###Print the last 100 lines of a container’s logs
```docker container logs --tail 100 web```

## create private registry

`docker run -d -p 5000:5000 --name registry registry:2`

tag a image to repo `docker image tag my-image localhost:5000/my-image`

`docker push localhost:5000/my-image`

`docker pull localhost:5000/my-image`

## remote docker

`docker -H=remote-docker-engine:2375 run nginx`

## cgroups

`docker run --cpus=.5 --memory=100m ubuntu`

