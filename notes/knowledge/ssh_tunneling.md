##### Local Port Forwarding: Make remote resources accessible on your local system
```bash
ssh -f -L2000:192.168.10.11:22 seliics00608 -N

ssh -p 2000 root@localhost
```


`-f` tells ssh to go into the background just before it executes the command
`-L` is in the form of *-L local-port:remote-host:remote-port*.
`-N` instructs OpenSSH to not execute a command on the remote system

`seliics00608` is the ssh jump server, `192.168.10.11` is the ip address that jump server can access

`root` is the username at `192.168.10.11`

if `seliics00608` uses a different user, then the first command becomes `ssh -f -L2000:192.168.10.11:22 root@seliics00608 -N`


##### Remote Port Forwarding: Make local resources accessible on a remote system
```bash
ssh -R 8888:localhost:1234 bob@ssh.youroffice.com
```

Someone could then connect to the SSH server at port 8888 and that connection would be tunneled to the server application runnig at port 1234 on the local PC


##### Dynamic Port Forwarding: Use your SSH server as a proxy
```bash
ssh -D 8888 bobo@ssh.yourhome.co
```
You could then configure a web browser or another application to use your local IP adddress(127.0.0.1) and port 8888. All traffic from the application would be redirected through the tunnel

