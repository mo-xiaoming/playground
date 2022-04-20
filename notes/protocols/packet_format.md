#### Ethernet Frame Format

<table border="1">
    <tr>
        <td>PREAMBLE</td>
        <td>SFD</td>
        <td>DST MAC</td>
        <td>SRC MAC</td>
        <td>LEN/EtherType</td>
        <td>PAYLOAD</td>
        <td>CRC/FCS</td>
    </tr>
    <tr>
        <td>7 bytes</td>
        <td>1 byte</td>
        <td>6 bytes</td>
        <td>6 bytes</td>
        <td>2 bytes</td>
        <td>46-1500 bytes</td>
        <td>4 bytes</td>
    </tr>
    <tr>
        <td>physical layer</td>
        <td>physical layer</td>
        <td>data link layer</td>
        <td>data link layer</td>
        <td>data link layer</td>
        <td>data link layer</td>
        <td>physical layer</td>
    </tr>
</table>

The preamble sequence followed by the SFD would be 0x55 0x55 0x55 0x55 0x55 0x55 0x55 0xD5

If `LEN` <=**1500 (0x5dc)** means that is is used to indicate the size of the payload; The values of **1536 (0x600)** and above indicate that is is used as an `EtherType` (as Ethernet II), to indicate which protocol is encapsulated in the payload of the frame, the length of the frame is determined by the locateion of the `interpacket gap` and valid `frame check sequence (FCS)`

`interpacket gap` is idel time between packets. After a packet has been sent, transmitters are required to transmit a minimum of **96 bits** of idle line state before transmitting the next packet.

##### Ethernet II Framing (DIX Ethernet)

`EtherType` == 0x0800 signals that the frame contains an IPv4 datagram.

`EtherType` == 0x0806 signals that the frame contains an ARP datagram.

`EtherType` == 0x08DD signals that the frame contains an IPv6 datagram.

##### Maximum througput

Protocol_efficiency = Payload_size / Packet_size

Maximum efficiency is achieved with largest allowed payload size and is:

1500/1538 = 97.53%

The packet size is maximum 1500 octet payload + 8 octet preamble + 14 octet header + 4 octet trailer + minimum interpacket gap 12 octet = 1538 octet

#### IP Header

<table border="1">
    <tr>
        <td>Offsets</td>
        <td colspan="4">F</td> <td colspan="4">F</td> <td colspan="4">F</td> <td colspan="4">F</td> <td colspan="4">F</td> <td colspan="4">F</td> <td colspan="4">F</td> <td colspan="4">F</td>
    </tr>
    <tr>
        <td>Octet</td>
        <td>0</td> <td>1</td> <td>2</td> <td>3</td> <td>4</td> <td>5</td> <td>6</td> <td>7</td>
        <td>0</td> <td>1</td> <td>2</td> <td>3</td> <td>4</td> <td>5</td> <td>6</td> <td>7</td>
        <td>0</td> <td>1</td> <td>2</td> <td>3</td> <td>4</td> <td>5</td> <td>6</td> <td>7</td>
        <td>0</td> <td>1</td> <td>2</td> <td>3</td> <td>4</td> <td>5</td> <td>6</td> <td>7</td>
    </tr>
    <tr>
        <td>0</td>
        <td colspan="4">Version</td>
        <td colspan="4">IHL</td>
        <td colspan="6">DSCP</td>
        <td colspan="2">ECN</td>
        <td colspan="16">Total Length</td>
    </tr>
    <tr>
        <td>4</td>
        <td colspan="16">Identification</td>
        <td colspan="3">Flags</td>
        <td colspan="13">Fragment Offset</td>
    </tr>
    <tr>
        <td>8</td>
        <td colspan="8">Time To Live</td>
        <td colspan="8">Protocol</td>
        <td colspan="16">Header Checksum</td>
    </tr>
    <tr>
        <td>12</td>
        <td colspan="32">Source IP</td>
    </tr>
    <tr>
        <td>16</td>
        <td colspan="32">Destination IP</td>
    </tr>
    <tr>
        <td>20</td>
        <td colspan="32" rowspan="4">Options (if IHL > 5)</td>
    </tr>
    <tr><td>24</td></tr>
    <tr><td>28</td></tr>
    <tr><td>32</td></tr>
</table>

`Version`, for IPv4, this is always equal to 4

`Internet Header Length (IHL)` specifies the size of the IPv4 header in 32-bit words. The value is between 5~15, that is 20~60 bytes

`Total Length` defines the entire IP packet size in bytes, including header and data. Size between 20 (header only)~65,535 bytes

`Identification` is primarily used for uniquely identifying the group of fragments of a single IP datagram

`Flags`

   1. bit 0: must be zero
   2. bit 1: Don't Fragment (DF)
   3. bit 2: More Fragments (MF)

`Fragment Offset` is measured in units of 8-byte blocks, it is 13 bits long and specifies the offset of a particular fragment relative the beginning of the original unfragmented IP datagram. This allows a maximum offset of (2^13 -1) x8 = 65,528 bytes.

`Time To Live (TTL)` hop count

`Protocol`

   1. 0x01: ICMP
   1. 0x06: TCP
   1. 0x11: UDP
   1. 0x29: IPv6

`Header Checksum` checksum of IP header

#### UDP Header
<table border="1">
    <tr>
        <td>Offsets</td>
        <td colspan="4">F</td> <td colspan="4">F</td> <td colspan="4">F</td> <td colspan="4">F</td> <td colspan="4">F</td> <td colspan="4">F</td> <td colspan="4">F</td> <td colspan="4">F</td>
    </tr>
    <tr>
        <td>Octet</td>
        <td>0</td> <td>1</td> <td>2</td> <td>3</td> <td>4</td> <td>5</td> <td>6</td> <td>7</td>
        <td>0</td> <td>1</td> <td>2</td> <td>3</td> <td>4</td> <td>5</td> <td>6</td> <td>7</td>
        <td>0</td> <td>1</td> <td>2</td> <td>3</td> <td>4</td> <td>5</td> <td>6</td> <td>7</td>
        <td>0</td> <td>1</td> <td>2</td> <td>3</td> <td>4</td> <td>5</td> <td>6</td> <td>7</td>
    </tr>
    <tr>
        <td>0</td>
        <td colspan="16">Source port</td>
        <td colspan="16">Destination port</td>
    </tr>
    <tr>
        <td>4</td>
        <td colspan="16">Length</td>
        <td colspan="16">Checksum</td>
    </tr>
</table>

`Length` is UPD header and data length in bytes, between 8~65,535 bytes (8 bytes header + 65,527 bytes of data) theoretically. However, the actual limit for the data length imposed by the under lying IPv4 protocol, is 65,507 bytes (65,535 - 8 byte UDP header - 20 byte IP header)

#### DNS HEADER
<table border="1">
    <tr>
        <td>Offsets</td>
        <td colspan="8">FF</td> <td colspan="8">FF</td> <td colspan="8">FF</td> <td colspan="8">FF</td>
    </tr>
    <tr>
        <td>Octet</td>
        <td>0</td> <td>1</td> <td>2</td> <td>3</td> <td>4</td> <td>5</td> <td>6</td> <td>7</td>
        <td>0</td> <td>1</td> <td>2</td> <td>3</td> <td>4</td> <td>5</td> <td>6</td> <td>7</td>
        <td>0</td> <td>1</td> <td>2</td> <td>3</td> <td>4</td> <td>5</td> <td>6</td> <td>7</td>
        <td>0</td> <td>1</td> <td>2</td> <td>3</td> <td>4</td> <td>5</td> <td>6</td> <td>7</td>
    </tr>
    <tr>
        <td>0</td>
        <td colspan="8">Opcode (1: REQ 2: REPLY)</td>
        <td colspan="8">Hardware type (1: Ethernet)</td>
        <td colspan="8">Hardware address length</td>
        <td colspan="8">Hop count</td>
    </tr>
    <tr>
        <td>4</td>
        <td colspan="32">Transcation ID</td>
    </tr>
    <tr>
        <td>8</td>
        <td colspan="16">Number of seconds</td>
        <td colspan="16">Flags (0bit: broadcasting)</td>
    </tr>
    <tr>
        <td>12</td>
        <td colspan="32">Client IP address</td>
    </tr>
    <tr>
        <td>16</td>
        <td colspan="32">Your IP address</td>
    </tr>
    <tr>
        <td>20</td>
        <td colspan="32">Server IP address</td>
    </tr>
    <tr>
        <td>24</td>
        <td colspan="32">Gateway IP address</td>
    </tr>
    <tr>
        <td>28</td>
        <td colspan="32" rowspan="4">Client Hardware Address</td>
    </tr>
    <tr>
        <td>32</td>
    </td>
    <tr>
        <td>36</td>
    </td>
    <tr>
        <td>40</td>
    </td>
    <tr>
        <td>44-107</td>
        <td colspan="32">Server host name</td>
    </tr>
    <tr>
        <td>108-235</td>
        <td colspan="32">Boot filename</td>
    </tr>
    <tr>
        <td>236</td>
        <td colspan="32">Options</td>
    </tr>
</table>

#### DHCP Header
<table border="1">
    <tr>
        <td>Offsets</td>
        <td colspan="4">F</td> <td colspan="4">F</td> <td colspan="4">F</td> <td colspan="4">F</td> <td colspan="4">F</td> <td colspan="4">F</td> <td colspan="4">F</td> <td colspan="4">F</td>
    </tr>
    <tr>
        <td>Octet</td>
        <td>0</td> <td>1</td> <td>2</td> <td>3</td> <td>4</td> <td>5</td> <td>6</td> <td>7</td>
        <td>0</td> <td>1</td> <td>2</td> <td>3</td> <td>4</td> <td>5</td> <td>6</td> <td>7</td>
        <td>0</td> <td>1</td> <td>2</td> <td>3</td> <td>4</td> <td>5</td> <td>6</td> <td>7</td>
        <td>0</td> <td>1</td> <td>2</td> <td>3</td> <td>4</td> <td>5</td> <td>6</td> <td>7</td>
    </tr>
    <tr>
        <td>0</td>
        <td colspan="16">Identification</td>
        <td colspan="1">Q/R</td>
        <td colspan="4">Opcode</td>
        <td colspan="1">AA</td>
        <td colspan="1">TC</td>
        <td colspan="1">RD</td>
        <td colspan="1">RA</td>
        <td colspan="1">Z</td>
        <td colspan="1">AD</td>
        <td colspan="1">CD</td>
        <td colspan="4">Rcode</td>
    </tr>
    <tr>
        <td>4</td>
        <td colspan="16">Total Questions</td>
        <td colspan="16">Total Answer RRs</td>
    </tr>
    <tr>
        <td>8</td>
        <td colspan="16">Total Authority RRs</td>
        <td colspan="16">Total Additional RRs</td>
    </tr>
    <tr>
        <td>12</td>
        <td colspan="32" style="background-color:lightgreen">Queries[]/Questions[]</td>
    </tr>
    <tr>
        <td></td>
        <td colspan="32" style="background-color:lightgreen">Answer RRs[]</td>
    </tr>
    <tr>
        <td></td>
        <td colspan="32" style="background-color:lightgreen">Authority RRs[]</td>
    </tr>
    <tr>
        <td></td>
        <td colspan="32" style="background-color:lightgreen">Additional RRs[]</td>
    </tr>
</table>

`QR` 0: Query; 1: Response

`Opcode`

|Opcode|Description|
|------|-----------|
|0 | QUERY |
|1 |IQUERY, Inverse query|
|2 |STATUS |
|4| Notify |
|5|Update|

`AA` Authoritative Answer

`TC`, Truncated, only the first 512 bytes of the reply was returned

`RD`, Recursion Desired

`RA`, Recursion Available

`AD`, Authenticated data

`CD`, Checking Disabled

`Rcode`, Return code

|Rcode|Description|
|-|-|
|0|No error|
|1|Format error|
|2|Server failure|
|3|Name Error|
|4|Not Implemented|
|5|Refused|
|6|YZDomain, Name Exists when it should not|
|7|YZRRSet, RR Set Exists when it should not|
|8|NXRRSet, RR Set that should exist does not|
|9|NotAuth, Sever Not authoritative for zone|
|10|NotZone, Name not contained in zone|
|16|BADVERS, Bad OPT Version; BADSIG, TSIG Signature Failure|
|17|BADKEY|
|18|BADTIME|
|19|BADMODE, Bad TKEY Mode|
|20|BADNAME, Duplicated key name|
|21|BADALG, Algorithm not supported|
|22|BADTRUNC, Bad truncation|

##### `Query`/`Question`

`Query Name`: variable length, ended by `0x00`

`Type`: 16 bits

|Type|Description|
|-|-|
|1|A, IPv4 address|
|2|NS, Authoritative name server|
|12|PTR, Domain name pointer|
|250|TSIG|

`Class` 16 bits

|Class|Description|
|-|-|
|1|IN, Internet|
|255|Any|