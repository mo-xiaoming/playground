## Server tcp
`nc -l 8080`

## Client tcp
`nc localhost 8000`

## Server UDP
`nc -l -u 8080`

## Client UDP
`nc -u localhost 8080`

## Setup a Server to recieve file
`nc -l 9999 > fromMac.file`

## Setup a client to Send
`nc 172.20.1.168 9999 < toLinux.file`

## If any ports between 1-100 is open
`nc -z -v -n 127.0.0.1 1-100`
