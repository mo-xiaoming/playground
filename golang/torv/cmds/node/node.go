package main

import (
	"flag"
	"log"
	"net/url"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gorilla/websocket"
)

func main() {
	master := flag.String("master", "", "master domain:port. <required>")
	node := flag.String("domain:port", "", "domain and CDN service port of this node. <required>")
	flag.Parse()

	if *node == "" || *master == "" {
		flag.PrintDefaults()
		os.Exit(1)
	}

	sigCh := make(chan os.Signal)
	signal.Notify(sigCh, os.Interrupt, syscall.SIGTERM)

	u := url.URL{Scheme: "ws", Host: *master, Path: "/ws"}
	for {
		if talk2Master(u, *node, sigCh) {
			return
		}
		time.Sleep(10 * time.Second)
	}
}

const (
	pingWait = 10 * time.Second
)

type quit bool

func talk2Master(u url.URL, node string, sigCh chan os.Signal) quit {
	log.Printf("(re)connecting to %s", u.String())

	ws, _, err := websocket.DefaultDialer.Dial(u.String(), nil)
	if err != nil {
		log.Printf("dial: %v", err)
		return false
	}
	defer func() {
		_ = ws.Close()
	}()

	ticker := time.NewTicker(pingWait)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			_ = ws.SetWriteDeadline(time.Now().Add(pingWait))
			err := ws.WriteMessage(websocket.TextMessage, []byte(node))
			if err != nil {
				log.Printf("write: %v", err)
				return false
			}
		case <-sigCh:
			return true
		}
	}
}
