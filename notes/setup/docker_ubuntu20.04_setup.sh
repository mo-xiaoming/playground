sudo apt install apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io

#sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker

cat <<EOF>>/etc/docker/daemon.json
{
    "registry-mirrors": [
        "https://cojcu4vl.mirror.aliyuncs.com",
        "http://registry.docker-cn.com"
    ]
}
EOF

#RUN  sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list

