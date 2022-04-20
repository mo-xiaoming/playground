package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"github.com/mo-xiaoming/torv/internal/httputils"
)

func main() {
	port := flag.Int("port", 0, "port this service is on. <required>")
	host := flag.String("host", "", "public ip of this service. <required>")
	redirectPort := flag.Int("redirectPort", 0, "redirect port. <required>")
	flag.Parse()

	if *port <= 0 || *host == "" || *redirectPort <= 0 {
		flag.PrintDefaults()
		os.Exit(1)
	}

	var nodeIPPool = ipPool{
		ips:       map[string]aliveness{},
		newIPChan: make(chan string),
		ipRemoved: make(chan string),
		done:      make(chan struct{}),
		m:         sync.Mutex{},
	}
	go nodeIPPool.start()

	var clientIPPool = clientIPPool {
		pool: map[string]clientIPAliveness{},
	}

	srv := http.Server{
		Addr:         fmt.Sprintf(":%d", *port),
		WriteTimeout: time.Second * 15,
		ReadTimeout:  time.Second * 15,
		IdleTimeout:  time.Second * 15,
		Handler:      setupRouter(&nodeIPPool, &clientIPPool, *host, *redirectPort),
	}

	go func() {
		if err := srv.ListenAndServe(); err != http.ErrServerClosed {
			log.Printf("starting on %s failed", srv.Addr)
			os.Exit(1)
		}
	}()
	log.Printf("server started on %s", srv.Addr)

	httputils.ReadyForInterruptSignal(&srv, 5*time.Second, nodeIPPool.stop)
}

func setupRouter(nodeIPPool *ipPool, clientIPPool *clientIPPool, host string, redirectPort int) *http.ServeMux {
	http.HandleFunc("/ws", ws(nodeIPPool))
	http.HandleFunc("/", redirect(nodeIPPool, clientIPPool, host, redirectPort))
	return http.DefaultServeMux
}

func ws(nodeIPPool *ipPool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var upgrader = httputils.NewWSUpgrader()
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			log.Printf("failed to set websocket upgrade: %+v", err)
			return
		}
		defer func() {
			log.Print("closing ws")
			if err := conn.Close(); err != nil {
				log.Print("cannot close ws properly")
			}
		}()

		var ip = ""
		if ip, _, err = httputils.GetIP(r); err != nil {
			log.Printf("failed to retrieve peer ip")
		}

		go nodeIPPool.incomingIP(ip)

		for {
			_ = conn.SetReadDeadline(time.Now().Add(60*time.Second))
			_, msg, err := conn.ReadMessage()
			if err != nil {
				go nodeIPPool.removeIP(ip)
				if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway) {
					log.Printf("error: %v, user-agent: %v", err, r.Header.Get("User-Agent"))
					break
				}
				log.Printf("ws closing: %v", err)
				break
			}
			log.Printf("got %v", string(msg))

			go nodeIPPool.incomingIP(ip)
		}
	}
}

func redirect(nodeIPPool *ipPool, clientIPPool *clientIPPool, host string, redirectPort int) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var ip string
		var err error
		if ip, _, err = httputils.GetIP(r); err != nil {
			log.Printf("redirect GetIP %v", err)
			http.Redirect(w, r, fmt.Sprintf("http://%s:%d%s", host, redirectPort, r.URL.Path), http.StatusMovedPermanently)
			return
		}

		if g, err := clientIPPool.get(ip); err == nil {
			closetIP := nodeIPPool.closedIP(g)
			if closetIP != "" {
				http.Redirect(w, r, fmt.Sprintf("http://ec2-18-179-111-253.ap-northeast-1.compute.amazonaws.com:%d%s", redirectPort, r.URL.Path), http.StatusMovedPermanently)
				//http.Redirect(w, r, fmt.Sprintf("http://%s:%d%s", closetIP, redirectPort, r.URL.Path), http.StatusMovedPermanently)
				return
			}
		}
		log.Printf("something went wrong, use host ip")
		http.Redirect(w, r, fmt.Sprintf("http://%s:%d%s", host, redirectPort, r.URL.Path), http.StatusMovedPermanently)
	}
}
