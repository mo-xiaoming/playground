`tcpdump -SXXnnvvvtttt`
`tcpdump -Xnnvvvtttt`

* `-i any` : Listen on all interfaces just to see if you’re seeing any traffic.
* `-i eth0` : Listen on the eth0 interface.
* `-D` : Show the list of available interfaces
* `-n` : Don’t resolve hostnames.
* `-nn` : Don’t resolve hostnames or port names.
* `-q` : Be less verbose (more quiet) with your output.
* `-X` : Show the packet’s contents in both hex and ASCII.
* `-XX` : Same as -X, but also shows the ethernet header.
* `-v -vv -vvv` : Increase the amount of packet information you get back.
* `-c` : Only get x number of packets and then stop.
* `icmp` : Only get ICMP packets.
* `-s` : Define the snaplength (size) of the capture in bytes. Use -s0 to get everything, unless you are intentionally capturing less.
* `-S` : Print absolute sequence numbers.
* `-e` : Get the ethernet header as well.
* `-E` : Decrypt IPSEC traffic by providing an encryption key.
* `-tttt` : Given maximally human-readable timestamp output
* `-l` : Make stdout line buffered.
* `-A` : Output in ASCII

1. AND
`and` or `&&`

2. OR
`or` or `||`

3. EXCEPT
`not` or `!`

##### Everything on an interface
Looking at what's hitting your interface. Or get all interfaces with `-i any`
`tcpdump -i eth0`

| \#         | \#                                |
|------------|-----------------------------------|
| Types      | `host`, `net`, `port`             |
| Directions | `src`, `dst`                      |
| Protocols  | `tcp`, `udp`, `icmp`, `arp`, `ip` |

##### Find traffic by IP
Traffic that's going to or from `1.1.1.1`
`tcpdump host 1.1.1.1`

##### Filtering by Source and/or Destination
`tcpdump src 1.1.1.1`
`tcpdump dst 1.0.0.1`

##### Finding packets by network
To find packets going to or from a paticular network or subnet
`tcpdump net 1.2.3.0/24`

##### Show traffic related to a specific port
`tcpdump port 3389`
`tcpdump src port 1025`

##### Show traffic of one protocol
`tcpdump icmp`

##### Show only IP6 traffic
`tcpdump ip6`

##### Find traffic using port ranges
`tcpdump portrange 21-23`

##### Find traffic based on packet size
`tcpdump less 32`
`tcpdump greater 64`
`tcpdump <= 128`

##### Reading/Writing caputres to a File (pcap)
Note that you can use all the regular commands within `tcpdump` wile reading in a file.
`tcpdump port 80 -w capture_file`
`tcpdump -r capture_file`
* `tcpdump host 1.2.3.4`
* `tcpdump src 2.3.4.5`
* `tcpdump dst 3.4.5.6`
* `tcpdump net 1.2.3.0/24`
* `tcpdump icmp`
* `tcpdump ip6`
* `tcpdump port 3389`
* `tcpdump src port 1025`
* `tcpdump dst port 389`
* `tcpdump src port 1025 and tcp`
* `tcpdump udp and src port 53`
* `tcpdump portrange 21-23`
* `tcpdump less 32` 
* `tcpdump greater 128~`
* `tcpdump <= 128`

### An mnemonic for the TCP flags: Unskilled Attackers Pester Real Security Folk

| \#        | Flag | Value | Meaning                                     |
|-----------|------|-------|---------------------------------------------|
| Unskilled | URG  | 32    | Urgent                                      |
| Attachers | ACK  | 16    | Ackowledges recieved data                   |
| Pester    | PSH  | 8     | Push                                        |
| Real      | RST  | 4     | Aborts a connection in response to an error |
| Security  | SYN  | 2     | Initiates a connection                      |
| Folks     | FIN  | 1     | Closes a connection                         |


##### Show me all URGENT (URG) packets...
`tcpdump 'tcp[13] & 32!=0'`
`tcpdump 'tcp[tcpflags] == tcp-urg'`

##### Show me all ACKNOWLEDGE (ACK) packets...
`tcpdump 'tcp[13] & 16!=0'`
`tcpdump 'tcp[tcpflags] == tcp-ack'`

##### Show me all PUSH (PSH) packets...
`tcpdump 'tcp[13] & 8!=0'`
`tcpdump 'tcp[tcpflags] == tcp-psh'`

##### Show me all RESET (RST) packets...
`tcpdump 'tcp[13] & 4!=0'`
`tcpdump 'tcp[tcpflags] == tcp-rst'`

##### Show me all SYNCHRONIZE (SYN) packets...
`tcpdump 'tcp[13] & 2!=0'`
`tcpdump 'tcp[tcpflags] == tcp-syn'`

##### Show me all FINISH (FIN) packets...
`tcpdump 'tcp[13] & 1!=0'`

##### Show me all SYNCHRONIZE && ACKNOWLEDGE (SYNACK) packets...
`tcpdump 'tcp[13]=18'`

##### Show me all SYNC || ACK packets...
`tcpdump 'tcp[tcpflags] & tcp-syn != 0 and tcp[tcpflags] & tcp-ack != 0'`

##### From specific IP and destined for a specific Port
Find all traffic from 10.5.2.3 going to any host on port 3389
`tcpdump -nnvvS src 10.5.2.3 and dst port 3389`

##### From on network to another
Look for all traffic coming from `192.168.x.x.` and going to the `10.x` or `172.16.x.x` networks, and we're showing hex output with no hostname resolution and one level of extra verbosity
`tcpdump -nvX src net 192.168.0.0/16 and dst net 10.0.0.0/8 or 172.16.0.0/16`

##### Non ICMP traffic going to a specific IP
Show all traffic going to `192.168.0.2` that is *not* ICMP
`tcpdump dst 192.168.0.2 and src net and not icmp`

##### Traffic from a host that isn't on a specific port
Show all traffic from a host that isn't SSH traffic
`tcpdump -vv src mars and not dst port 22`

##### Find HTTP User Agents
`tcpdump -vvAls0 |grep 'User-Agent:'`


`tcpdump 'src 10.0.2.4 and (dst port 3389 or 22)`

##### Find SMTP packets
`0x4d41494c` is "MAIL"
`tcpdump '((port 25) and (tcp[(tcp[12]>>2):4] = 0x4d41494c))'`

##### Find HTTP GET data
`0x47455420` is "GET "
`tcpdump 'tcp[(tcp[12]>>2):4] = 0x47455420'`

##### Find SSH data
`0x5353482D` is "SSH-"
`tcpdump 'tcp[(tcp[12]>>2):4] = 0x5353482D'`

##### Find DNS request
`tcpdump port 53`

##### Find FTP traffic
`tcpdump port ftp or ftp-data`

##### Find cleartext password
`tcpdump port http or port http or port smtp or port imap or port pop3 or port telnet -lA |egrep -i -B5 'pass=|pwd=|log=|login=|user=|username=|pw=|passw=|passwd= |password=|pass:|user:|username:|password:|login:|pass |user '`
