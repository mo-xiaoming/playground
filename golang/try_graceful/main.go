package main

import (
	"context"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"
)

func main() {
	interruptChan := make(chan os.Signal)
	signal.Notify(interruptChan, os.Interrupt, syscall.SIGTERM)

	ctx, cancel := context.WithCancel(context.Background())

	go func() {
		sig := <-interruptChan
		log.Println("received signal:", sig)
		cancel()
	}()

	go worker(ctx)
	time.Sleep(13 * time.Second)
}

func worker(ctx context.Context) {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	select {
	case <-ctx.Done():
		log.Println("canceling")
		return
	case <-ticker.C:
		log.Println("ticking")
	}
}
