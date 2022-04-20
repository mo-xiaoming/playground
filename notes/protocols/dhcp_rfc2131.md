#### three mechanisms for IP address allocatioin

1. *automatic allocation*: DHCP assigns a permanent IP address to a client

2. *dynamic allocation*: DHCP assigns an IP address to a client for a limited period of time

3. *manual allocation*: a client's IP address is assigned by the network administrator, and DHCP is sued to convey the assigned address to the client


#### updates from RFC1541

1. new message type, DHCPINFORM

2. "vendor" classes

3. the minimum lease time restriction has been removed


#### differences between DHCP and BOOTP

1. DHCP defines mechanisms through which clients can be assigned a network assdress for a finite lease, allowing for serial reassgiment of network addresses to different clients.

2. DHCP provides the mechanism for a client to acquire all of the IP configuration parameters that it needs in order to operate

```
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|     op (1)    |   htype (1)   |   hlen (1)    |   hops (1)    |
+---------------+---------------+---------------+---------------+
|                            xid (4)                            |
+-------------------------------+-------------------------------+
|           secs (2)            |           flags (2)           |
+-------------------------------+-------------------------------+
|                          ciaddr  (4)                          |
+---------------------------------------------------------------+
|                          yiaddr  (4)                          |
+---------------------------------------------------------------+
|                          siaddr  (4)                          |
+---------------------------------------------------------------+
|                          giaddr  (4)                          |
+---------------------------------------------------------------+
|                                                               |
|                          chaddr  (16)                         |
|                                                               |
|                                                               |
+---------------------------------------------------------------+
|                                                               |
|                          sname   (64)                         |
+---------------------------------------------------------------+
|                                                               |
|                          file    (128)                        |
+---------------------------------------------------------------+
|                                                               |
|                          options (variable)                   |
+---------------------------------------------------------------+
```

```
FIELD      OCTETS       DESCRIPTION
-----      ------       -----------

op            1  Message op code / message type.
				1 = BOOTREQUEST, 2 = BOOTREPLY
htype         1  Hardware address type, see ARP section in "Assigned
				Numbers" RFC; e.g., '1' = 10mb ethernet.
hlen          1  Hardware address length (e.g.  '6' for 10mb
				ethernet).
hops          1  Client sets to zero, optionally used by relay agents
				when booting via a relay agent.
xid           4  Transaction ID, a random number chosen by the
				client, used by the client and server to associate
				messages and responses between a client and a
				server.
secs          2  Filled in by client, seconds elapsed since client
				began address acquisition or renewal process.
flags         2  Flags (see figure 2).
ciaddr        4  Client IP address; only filled in if client is in
				BOUND, RENEW or REBINDING state and can respond
				to ARP requests.
yiaddr        4  'your' (client) IP address.
siaddr        4  IP address of next server to use in bootstrap;
				returned in DHCPOFFER, DHCPACK by server.
giaddr        4  Relay agent IP address, used in booting via a
				relay agent.
chaddr       16  Client hardware address.
sname        64  Optional server host name, null terminated string.
file        128  Boot file name, null terminated string; "generic"
				name or null in DHCPDISCOVER, fully qualified
				directory-path name in DHCPOFFER.
options     var  Optional parameters field.  See the options
				documents for a list of defined options.
```

>if the DHCP server and the DHCP client
>are connected to the same subnet (i.e., the 'giaddr' field in the
>message from the client is zero), the server SHOULD select the IP
>address the server is using for communication on that subnet as the
>'server identifier'.  If the server is using multiple IP addresses on
>that subnet, any such address may be used.  If the server has
>received a message through a DHCP relay agent, the server SHOULD
>choose an address from the interface on which the message was
>recieved as the 'server identifier' (unless the server has other,
>better information on which to make its choice).  DHCP clients MUST
>use the IP address provided in the 'server identifier' option for any
>unicast requests to the DHCP server.
`0x0044`: giaddr

>DHCP messages broadcast by a client prior to that client obtaining
>its IP address must have the source address field in the IP header
>set to 0.

>If 'giaddr' is zero and 'ciaddr' is zero, and the broadcast bit is
>set, then the server broadcasts DHCPOFFER and DHCPACK messages to
>0xffffffff
`0ac6 1b04`

