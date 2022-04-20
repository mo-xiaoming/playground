unzip -p 2020-08-20-raspios-buster-armhf.zip | sudo dd of=/dev/sdX bs=4M conv=fsync status=progress

sources.list

deb http://mirrors.aliyun.com/raspbian/raspbian/ buster main contrib non-free rpi

source.list.d/raspi.list

deb http://mirrors.ustc.edu.cn/archive.raspberrypi.org/debian/ buster main

> /boost/ssh
cat -> /boot/wpa_supplicant.conf

country=CN
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1

network={
scan_ssid=1
ssid="R400"
psk="trustnoone"
}

cat ->> /boot/config.txt

framebuffer_width=1024
framebuffer_height=768

curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker Pi
docker run hello-world
