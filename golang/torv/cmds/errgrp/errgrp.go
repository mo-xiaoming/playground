package main

import (
	"context"
	log "github.com/sirupsen/logrus"
	"golang.org/x/sync/errgroup"
	"io/ioutil"
	"net/http"
	"net/http/httptrace"
	"os"
	"os/signal"
	"syscall"
	"time"
)

type query struct {
	url  string
	resp string
	err  error
}

var (
	trace = &httptrace.ClientTrace{
		DNSDone: func(dnsInfo httptrace.DNSDoneInfo) {
			log.Tracef("DNS Done: %+v", dnsInfo)
		},
		GotConn: func(connInfo httptrace.GotConnInfo) {
			log.Tracef("Got Conn: %+v", connInfo)
		},
		ConnectStart: func(network, addr string) {
			log.Tracef("Dial start %s %s", network, addr)
		},
		ConnectDone: func(network, addr string, err error) {
			log.Tracef("Dial done %s %s, error: %+v", network, addr, err)
		},
	}
)

func search(ctx context.Context, q query, resultChan chan<- query) func() error {
	return func() error {
		reqCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
		defer cancel()

		req, err := http.NewRequest(http.MethodGet, q.url, nil)
		if err != nil {
			q.err = err
			resultChan <- q
			return nil
		}

		req = req.WithContext(httptrace.WithClientTrace(reqCtx, trace))
		//req = req.WithContext(reqCtx)
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			q.err = err
			resultChan <- q
			return nil
		}

		body, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			q.err = err
			resultChan <- q
			return nil
		}
		_ = resp.Body.Close()

		q.resp = string(body)
		resultChan <- q
		return nil
	}
}

func fetch(ctx context.Context, queries []query, resultChan chan<- query) error {
	defer close(resultChan)

	eg, egCtx := errgroup.WithContext(ctx)
	for _, q := range queries {
		eg.Go(search(egCtx, q, resultChan))
	}

	return eg.Wait()
}

func setupLogger() {
	log.SetFormatter(&log.JSONFormatter{
		TimestampFormat:   time.RFC3339Nano,
		DisableHTMLEscape: true,
		DataKey:           "data",
	})
	log.SetLevel(log.TraceLevel)
	log.WithFields(log.Fields{
		"related": "errGroup", "function": "context cancellation",
	}).Info("error group example logs")
}

func main() {
	setupLogger()

	termChan := make(chan os.Signal)
	signal.Notify(termChan, syscall.SIGTERM, syscall.SIGINT)

	cxt, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() {
		log.Trace("waiting for shutdown signal")
		sig := <-termChan
		log.Tracef("sig %s accepted, shutdown sequence commenced", sig)
		cancel()
	}()

	var queries = []query{
		{url: "http://localhost:45678/fast"},
		{url: "http://localhost:45678/slow"},
	}
	var resultsChan = make(chan query, len(queries))
	if err := fetch(cxt, queries, resultsChan); err != nil {
		log.Errorf("get error %+v", err)
	} else {
		log.Trace("everything is fine")
	}
	for r := range resultsChan {
		if r.err != nil {
			log.Errorf("%s error: %+v", r.url, r.err)
			continue
		}
		log.Infof("%s result: %s", r.url, r.resp)
	}
}
