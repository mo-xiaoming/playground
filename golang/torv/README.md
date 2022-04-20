![Go](https://github.com/mo-xiaoming/torv/workflows/Go/badge.svg)

## on master

```bash
nohup ./master -host cnet-s5.wencaischool.net -port 11180 -redirectPort 4448 &

nohup ./fakecdn -dir . -listen :4448 &
```

## on node

```bash
nohup ./node -master cnet-s5.wencaischool.net:11180 -node node1 &

nohup ./fakecdn -dir . -listen :4448 &
```

## testing

```bash
curl http://cnet-s5.wencaischool.net/ispace2_upload5/71/2018-10-18/5872/content/ch010101/HD1-1-1.mp4 -o HD1-1-1.mp4
```
