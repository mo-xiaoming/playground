package main

import (
	"fmt"
	"net/http"
	"time"
)

func staticHandler(w http.ResponseWriter, r *http.Request) {
	http.ServeFile(w, r, "static/hello.html")
}

func dynamicHandler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "current time %+v", time.Now())
}

func main() {
	http.HandleFunc("/static", staticHandler)
	http.HandleFunc("/", dynamicHandler)
	http.ListenAndServe(":8080", nil)
}
