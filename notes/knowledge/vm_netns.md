#### Types of virtual network interfaces
* **Bridge**: A Linux bridge behaves like a network switch. It forwards packets between interfaces that are connected to it.
* **TUN**: TUN (network tunnel) devices work at the IP level or layer three level of the network stack and are usually point-to-point connections. A typical use for a TUN device is establishing VPN connections since it gives the VPN software a change to encrypt the data before it gets put on the wire. Since a TUN device works at layer three it can only accept IP packets and in some cases only IPv4.
* **TAP**: TAP (terminal access point) devices, in contrast, work at the Ethernet level or layer two and therefore behave very much like a real network adapter. Since they are running at layer two they can transport any layer three protocol and aren't limited to point-to-point connections.
* **VETH**: Virtual Ethernet interfaces are essentially a virtual equivalent of a patch calbe, what goes in one end comes out the other. When either device is donw, the link state of the pair is done

##### creating a bridge
```bash
ip link add br0 type bridge
```

##### enslaving a network interface to a bridge
```bash
ip link set eth0 master br0
```

##### an example of creating two virtual ethernet interfaces (veth1, veth2) and linking them together
```bash
ip link add veth1 type veth peer name veth2
```

##### `veth` interface can also be linked to a bridge
```bash
ip link set veth1 master br0
ip link set veth2 master br0
```

##### adding IP addresses to the interfaces
```bash
ip addr add 10.0.0.10 dev veth1
ip addr add 10.0.0.11 dev veth2
```

#### Network namespace allows different processes to have different views of the network and different aspects of networking can be isolated between processes:

* Interfaces: different processes can connect to addresses on different interfaces
* Routes: since processes can see different addresses from different namespaces, they also need different routes to connect to networks on those interfaces.
* Firewall rules: since there are dependant on the source or target interfaces, you may need different firewall rules in different network namespaces.


##### create, list and delete a network namespace
```bash
ip netns add ns1
ip netns list
ip netns del ns1
```

##### distinct network namespaces can be connected together using `veth` interfaces
```bash
ip netns add ns1
ip netns add ns2
ip link add veth1 netns ns1 type veth peer name veth2 netns ns2
```

##### `veth` interfaces can be assigned an IP address, inside a network class
```bash
ip netns exec ns1 ip addr add "10.0.0.1/24" dev veth1
ip netns exec ns2 ip addr add "10.0.0.2/24" dev veth2
```

##### once the IPs are assigned, the veth interfaces have to be brought in `UP` state
```bash
ip netns exec ns1 ip link set veth1 up
ip netns exec ns2 ip link set veth2 up
```

##### running `ping` command between the two different namespaces through the `veth` interfaces
```bash
ip nets exec ns1 ping -c 2 10.0.0.2

ip nets exec ns2 ping -c 2 10.0.0.1
```
##### a network namespace can have its own network interace assigned to it, for example the loopback interface (which is **by default** always present on new network NS but in `DOWN` state)
```bash
ip netns exec ns1 ip link set lo up
```

##### a network namespace can also have a separated routing table
```bash
ip netns exec ns1 ip route show
```

##### once a network NS is created, it will shows up in multiple places
```bash
mount |grep netns

ls -l /var/run/netns
```

#### A virtual network with network namespaces and a bridge

```
      br-Veth1          Veth1 +-------------+
         +--------------------+ namespace 1 |
         |                    +-------------+
+--------+
|        |
| bridge |
|        |
+--------+
         |              Veth2 +-------------+
         +--------------------+ namespace 2 |
      br-Veth2                +-------------+
```
* br-veth{1,2}: veth attached to the bridge
* veth{1,2}: veth part of their respective network NS

##### create two network NS
```bash
ip netns add ns1
ip netns add ns2
```

##### create two pairs of veth
```bash
ip link add veth1 type veth peer name br-veth1
ip link add veth2 type veth peer name br-veth2
```

##### attach to new veths to the network NS
```bash
ip link set veth1 netns ns1
ip link set veth2 netns ns2
```

##### assign IP addresses to `veth1` and `veth2`
```bash
ip netns exec ns1 ip addr add 192.168.1.11/24 dev veth1
ip netns exec ns2 ip addr add 192.168.1.11224 dev veth2
```

##### add a bridge allow two veth communicate between each other
```bash
ip link add name br1 type bridge
ip link set br1 up
```

##### connect the two veth interfaces (`br-veth{1,2}`) and attach them to the bridge
```bash
ip link set br-veth1 up
ip link set br-veth2 up
```

##### setup ip address for the bridge
```bash
ip addr add 192.168.1.10/24 brd + dev br1
```

Confirmed by checking routing table, `ip route`
   > 192.168.1.0/24 dev br1 proto kernel scope link src 192.168.1.10

##### ping 192.168.1.{11,12} from the global network NS ??

##### reach ns2 from ns1 by defining the proper routing
```bash
ip netns exec ns1 ip route 192.168.1.0/24 dev veth1 proto kernel scope link src 192.168.1.11
```

##### reach outside world by NAT through iptables
```bash
iptables -t nat -A POSTROUTING -s 192.168.1.0/24 -j MASQUERADE
```
specifies that on the NAT table, we are appending (`-A`) a rule to the `POSTROUTING` chain for the source address specified (`-s`) and the action will be `MASQUERADE`

##### enable IP forwarding
```bash
sysctl -w net.ipv4.ip_forward=1
```

confirmed by 
```bash
ip netns exec ns1 ping 8.8.8.8
```
