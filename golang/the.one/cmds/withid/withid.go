package main

import (
	"fmt"
	"net/http"
	"os"

	"github.com/gorilla/mux"
)

func pageHandler(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	pageID := vars["id"]
	fileName := fmt.Sprintf("static/%s.html", pageID)
	if _, err := os.Stat(fileName); err != nil {
		fileName = "static/404.html"
	}
	http.ServeFile(w, r, fileName)
}

func main() {
	router := mux.NewRouter()
	router.HandleFunc("/page/{id:[0-9]+}", pageHandler).Methods(http.MethodGet)
	http.Handle("/", router)
	http.ListenAndServe(":8080", nil)
}
