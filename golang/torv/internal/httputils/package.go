package httputils

import (
	"context"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"time"

	"github.com/gorilla/websocket"
)

func ReadyForInterruptSignal(srv *http.Server, duration time.Duration, extra func()) {
	interrupt := make(chan os.Signal)
	signal.Notify(interrupt, os.Interrupt)
	<-interrupt

	ctx, cancel := context.WithTimeout(context.Background(), duration)
	defer cancel()

	go extra()

	if err := srv.Shutdown(ctx); err != nil {
		log.Println("failed to shutdown")
		return
	}
	log.Println("shutting down")
}

func NewWSUpgrader() websocket.Upgrader {
	return websocket.Upgrader{
		ReadBufferSize:    1024,
		WriteBufferSize:   1024,
		EnableCompression: true,
	}
}

const (
	IPError = iota
	IPFromXRealIP
	IPFromXForwardFor
	IPFromRemoteAddr
)

func GetIP(r *http.Request) (string, int, error) {
	ip := r.Header.Get("X-REAL-IP")
	netIP := net.ParseIP(ip)
	if netIP != nil {
		return ip, IPFromXRealIP, nil
	}

	ips := r.Header.Get("X-FORWARD-FOR")
	splitIPs := strings.Split(ips, ",")
	for _, ip := range splitIPs {
		netIP := net.ParseIP(ip)
		if netIP != nil {
			return ip, IPFromXForwardFor, nil
		}
	}

	if ip, _, err := GetPeer(r); err == nil {
		return ip, IPFromRemoteAddr, nil
	}

	return "", IPError, fmt.Errorf("no valid client ip")
}

func GetPeer(r *http.Request) (string, int, error) {
	ip, port, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return "", 0, fmt.Errorf("%s is not a valid remote address", r.RemoteAddr)
	}
	netPort := 0
	if netPort, err = strconv.Atoi(port); err != nil {
		return "", 0, fmt.Errorf("%s is not a valid port number", port)
	}
	if net.ParseIP(ip) == nil {
		return "", 0, fmt.Errorf("%s is not a valid ip", ip)
	}
	return ip, netPort, nil
}
