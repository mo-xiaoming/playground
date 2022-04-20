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
		Addr:         fmt.Sprintf(":45678"),
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
	http.HandleFunc("/fast", func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(3*time.Second)
		_, _ = fmt.Fprint(w, "fast")
	})
	http.HandleFunc("/slow", func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(10*time.Second)
		_, _ = fmt.Fprint(w, "slow")
	})
	return http.DefaultServeMux
}
