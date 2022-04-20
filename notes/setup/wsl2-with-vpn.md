[github issue 4698](https://github.com/microsoft/WSL/issues/4698#issuecomment-814259640)

>A simple fix for me was to set MTU to 1350 (same as VPN interface): `sudo ifconfig eth0 mtu 1350`
>
>Even SSH connections are more stable now.

This solved my issues as well. I am using checkpoint VPN.

to get the MTU value of your VPN run below command for checkpoint, for example :

```
Get-NetAdapter | Where-Object {$_.InterfaceDescription -Match "Check"}

Name                      InterfaceDescription                    ifIndex Status                   LinkSpeed
----                      --------------------                    ------- ------                  ---------
Ethernet 3                Check Point Virtual Network Adapter Fo…       8 Up                   1 Gbps
```

Note the index of the connection, then run below statement. Match the index found above in below result and note the mtu.

```
netsh.exe interface ipv4 show interfaces

Idx     Met         MTU          State                Name
---  ----------  ----------  ------------  ---------------------------
  1          75  4294967295  connected     Loopback Pseudo-Interface 1
 24          35        1500  connected     Wi-Fi
 16          25        1500  disconnected  Local Area Connection* 1
 10           5        1500  disconnected  Ethernet 2
  5          25        1500  disconnected  Local Area Connection* 2
 17          65        1500  disconnected  Bluetooth Network Connection
 38          15        1500  connected     vEthernet (Default Switch)
 59          15        1500  connected     vEthernet (WSL)
  8           0        1350  connected     Ethernet 3
```

Not you have the mtu value.. Login into your wsl and run below statement(1350 is the mtu found above):

`sudo ifconfig eth0 mtu 1350`

You can also run this script when you wsl2 is starting. below is the script if you have debian based wsl

`sudo ip link set dev eth0 mtu 1350`
