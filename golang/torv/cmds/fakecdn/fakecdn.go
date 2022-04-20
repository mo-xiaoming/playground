package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"time"
)

func main() {
	port := flag.Int("port", -1, "port. <required>")
	directory := flag.String("dir", "", "files under this folder will be served. <required>")
	flag.Parse()

	if *port <= 0 || *directory == "" {
		flag.PrintDefaults()
		os.Exit(1)
	}

	http.Handle("/", http.FileServer(http.Dir(*directory)))
	srv := http.Server{
		Addr:         fmt.Sprintf(":%d", *port),
		WriteTimeout: time.Second * 15,
		ReadTimeout:  time.Second * 15,
		IdleTimeout:  time.Second * 15,
	}
	log.Printf("serving on %d", *port)
	log.Fatal(srv.ListenAndServe())
}
