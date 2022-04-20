package main

import (
	"fmt"
	"time"
)

const (
	logDebug = "DEBUG"
	logInfo = "INFO "
	logError = "ERROR"
)

type logEntry struct {
	when time.Time
	level string
	msg string
}

var logCh = make(chan logEntry, 50)

func logger() {
	for entry := range logCh {
		fmt.Printf("%v [%s] %s\n", entry.when.Format("2006-01-02T15:04:05.000000"), entry.level, entry.msg)
	}
}

func main() {
	go logger()

	logCh <-logEntry{time.Now(), logDebug, "hello"}
	logCh <-logEntry{time.Now(), logDebug, "world"}
}
