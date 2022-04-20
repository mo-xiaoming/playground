# scan one ip
nmap 192.168.0.100

# more info
nmap -v 192.168.0.100

# scan multiple ip
nmap 192.168.0.100 192.168.0.101 192.168.0.102
nmap 192.168.0.100,101,102

# scan a range of ip
nmap 192.168.0.1-20
nmap 192.168.0.*
nmap 192.168.0.0/24

# scan ip from file
nmap -iL filename

# exclude ip
nmap 192.168.1.0/24 --exclude 192.168.0.100,103
nmap -iL filename --excludefile exclude_filename

# os detection
nmap -A 192.168.0.100
nmap -v -A 192.168.0.100

# firewall protected
nmap -sA 192.168.0.100

# scan an ip behind a wall
nmap -PN 192.168.0.100

# scan running devices
nmap -sP 192.168.0.0/24

# reason in that state?
nmap --reason 192.168.0.100

# show all packets sent and received
nmap --packet-trace 192.168.0.100

# scan a port
nmap -p 80 192.168.0.100
nmap -p T:80 192.168.0.100 # tcp
nmap -p U:53 192.168.0.100 # udp
nmap -p 80,443 192.168.0.100 # multiple ports
nmap -p 80-200 192.168.0.100 # port ranges
nmap -v -sU -sT -p U:53,111,137,T:21-25,80,139,8080 192.168.0.100 # combine
nmap -p "*" 192.168.0.100 # all ports

# scan most common ports
nmap --top-ports 5 192.168.0.100

# quickest but aggressive scan
nmap -T5 192.168.0.0/24

# detect os
nmap -O 192.168.0.100
nmap -O --osscan-guess 192.168.0.100
nmap -v -O --osscan-guess 192.168.0.100

# service version number
nmap -sV 192.168.0.100

# scan using ACK SYC ping
nmap -PS 192.168.0.100
nmap -PS 80,21,443 192.168.0.100
nmap -PA 192.168.0.100
nmap -PA 80,21,200-512 192.168.0.100

# scan using ip ping
nmap -PO 192.168.0.100

# scan using udp ping
nmap -PU 192.168.0.100

# stealthy scan
nmap -sS 192.168.0.100

# scan for ip protocal
nmap -sO 192.168.0.100

# cloak a scan with decoy
nmap -n -D192.168.1.5,10.5.1.2,172.1.2.4,3.4.2.1 192.168.0.100
