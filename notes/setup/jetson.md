```
deb http://mirrors.ustc.edu.cn/ubuntu-ports/ bionic-updates main restricted
deb http://mirrors.ustc.edu.cn/ubuntu-ports/ bionic universe
deb http://mirrors.ustc.edu.cn/ubuntu-ports/ bionic-updates universe
deb http://mirrors.ustc.edu.cn/ubuntu-ports/ bionic multiverse
deb http://mirrors.ustc.edu.cn/ubuntu-ports/ bionic-updates multiverse
deb http://mirrors.ustc.edu.cn/ubuntu-ports/ bionic-backports main restricted universe multiverse
deb http://mirrors.ustc.edu.cn/ubuntu-ports/ bionic-security main restricted
deb http://mirrors.ustc.edu.cn/ubuntu-ports/ bionic-security universe
deb http://mirrors.ustc.edu.cn/ubuntu-ports/ bionic-security multiverse
```

## disable ui

```bash
sudo systemctl set-default multi-user.target
sudo reboot
```

## enable ui

```bash
sudo systemctl set-default graphical.target
sudo reboot
```
