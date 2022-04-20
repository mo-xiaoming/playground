package main

import (
	"fmt"
	"log"
	"net/http"
	"os"
	"time"
)

func main() {
	srv := http.Server{
		Addr:         fmt.Sprintf(":11180"),
		WriteTimeout: time.Second * 15,
		ReadTimeout:  time.Second * 15,
		IdleTimeout:  time.Second * 15,
		Handler:      setupRouter(),
	}

	if err := srv.ListenAndServe(); err != http.ErrServerClosed {
		log.Printf("starting on %s failed", srv.Addr)
		os.Exit(1)
	}
}

func setupRouter() *http.ServeMux {
	http.HandleFunc("/", redirect)
	return http.DefaultServeMux
}

func redirect(w http.ResponseWriter, r *http.Request) {
	http.Redirect(w, r, fmt.Sprintf("https://aws-node2.ddns.net:4448/%s", r.URL.Path), http.StatusMovedPermanently)
}
