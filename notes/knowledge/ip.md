##### Adding or Deleting an IP Address
`ip addr add 192.168.80.174 dev eth0`
`ip addr del 192.168.80.174 dev eth0`

##### Set MAC Hardware Address to Network Interface
`ip link set dev eth0 address 00:0c:29:33:4e:aa`

##### Set MTU value 2000
`ip link set dev eth0 mtu 2000`

##### Enable/Disable multipcast flag
`ip link set dev eth0 multicast`

##### Setting the transmit queue length
`ip link set dev eth0 txqueuelen 1200`

##### Enable/Disable promiscuous mode
`ip link set dev eth0 promisc on`

##### Enable/Disable all multicast mode
`ip link set dev eth0 allmulti on`

##### Enable/Disable network interface
`ip link set eth0 down`
`ip link set eth0 up`

##### Enable/Disable ARP Protocol
`ip link set dev eth0 arp on`

##### Add an entry in ARP table
`ip neigh add 192.168.0.1 lladdr 00:11:22:33:44:55 nud permanent dev eth0`

##### Add Static Route (gateway)?
`ip route add 10.10.20.0/24 via 192.168.50.100 dev eth0`

##### Add a route
`ip route add 192.168.3.0/24 dev eth3`

##### Delete entries from a routing table
`ip route del 192.168.3.0/24 dev eth3`

##### Which interface a packet to a given IP address would be route to
`ip route get 192.168.88.77`

##### Add alias interface
`ip addr add 10.0.0.1/8 dev eth0 label eth0:1`
