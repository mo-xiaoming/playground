https://askubuntu.com/questions/29284/how-do-i-mount-shared-folders-in-ubuntu-using-vmware-tools

```
Most other answers are outdated. For Ubuntu 18.04 (or recent Debian distros), try:

sudo vmhgfs-fuse .host:/ /mnt/hgfs/ -o allow_other -o uid=1000
If the hgfs directory doesn't exist, try:

sudo vmhgfs-fuse .host:/ /mnt/ -o allow_other -o uid=1000
You may have use a specific folder instead of .host:/. In that case you can find out the share's name with vmware-hgfsclient. For example:

$ vmware-hgfsclient
my-shared-folder
$ sudo vmhgfs-fuse .host:/my-shared-folder /mnt/hgfs/ -o allow_other -o uid=1000
If you want them mounted on startup, update /etc/fstab with the following:

# Use shared folders between VMWare guest and host
.host:/    /mnt/hgfs/    fuse.vmhgfs-fuse    defaults,allow_other,uid=1000     0    0
I choose to mount them on demand and have them ignored by sudo mount -a and the such with the noauto option, because I noticed the shares have an impact on VM performance.

Requirements
Software requirements may require installing the following tools beforehand:

sudo apt-get install open-vm-tools open-vm-tools-desktop
Others have claimed the following are required:

sudo apt-get install build-essential module-assistant \
  linux-headers-virtual linux-image-virtual && dpkg-reconfigure open-vm-tools

Can confirm, on Ubuntu 19.10 virtual machine running on Windows 10 Pro host, this worked. Specifically, sudo vmhgfs-fuse .host:/ /mnt/ -o allow_other -o uid=1000 –
Jason Shultz
 Mar 10 '20 at 18:23
 ```
