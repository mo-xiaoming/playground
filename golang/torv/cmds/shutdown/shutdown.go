package main

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	log "github.com/sirupsen/logrus"
)

func main() {
	log.SetLevel(log.TraceLevel)

	wg := &sync.WaitGroup{}

	srv := http.Server{
		Addr:         ":45678",
		WriteTimeout: time.Second * 15,
		ReadTimeout:  time.Second * 15,
		IdleTimeout:  time.Second * 60,
	}

	http.Handle("/", Adapt(http.HandlerFunc(worker), WithWaitGrp(wg), Log()))

	go StartServer(&srv)
	WaitForShutdown(wg, &srv)
}

func WaitForShutdown(wg *sync.WaitGroup, srv *http.Server) {
	wg.Add(1)
	go waitForShutdown(wg, srv)
	wg.Wait()
}

func waitForShutdown(wg *sync.WaitGroup, srv *http.Server) {
	defer wg.Done()

	termChan := make(chan os.Signal)
	signal.Notify(termChan, syscall.SIGTERM, syscall.SIGINT)

	sig := <-termChan
	log.Infof("signal %+v received. Shutdown sequence commenced", sig)
	if err := srv.Shutdown(context.Background()); err != nil {
		log.Errorf("shutdown error %+v", err)
	}
}

func StartServer(srv *http.Server) {
	if err := srv.ListenAndServe(); err != nil {
		if err == http.ErrServerClosed {
			log.Trace("server closed")
		} else {
			log.Errorf("listen and serve error %+v", err)
		}
	}
}

func worker(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	time.Sleep(5 * time.Second)
	_, _ = fmt.Fprint(w, "ok")
}

type Adapter func(http.Handler) http.Handler

func Adapt(h http.Handler, adapters ...Adapter) http.Handler {
	for _, adapter := range adapters {
		h = adapter(h)
	}
	return h
}

type StatusRecorder struct {
	http.ResponseWriter
	Status int
}

func (r *StatusRecorder) WriteHeader(status int) {
	r.Status = status
	r.ResponseWriter.WriteHeader(status)
}

func Log() Adapter {
	return func(h http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			recorder := &StatusRecorder{
				ResponseWriter: w,
				Status:         200,
			}
			start := time.Now()
			h.ServeHTTP(recorder, r)
			log.Infof("%s %d %s %dns %+[4]v", r.URL.Path, recorder.Status, r.RemoteAddr, time.Since(start))
		})
	}
}

func WithWaitGrp(wg *sync.WaitGroup) Adapter {
	return func(h http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			wg.Add(1)
			defer wg.Done()
			h.ServeHTTP(w, r)
		})
	}
}
