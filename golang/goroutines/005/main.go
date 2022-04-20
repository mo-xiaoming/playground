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

type logger struct {
	logCh chan logEntry
	discard bool
}

func (l *logger) init() {
	l.logCh = make(chan logEntry, 50)
	go func(l *logger) {
		for entry := range l.logCh {
			fmt.Printf("%v [%s] %s\n", entry.when.Format("2006-01-02T15:04:05.000000"), entry.level, entry.msg)
		}
	}(l)
}


func (l *logger) log(entry logEntry) {
	if !l.discard {
		l.logCh <-entry
	}
}

func (l *logger) setDiscard(discard bool) {
	l.discard = discard
}

func (l *logger) destroy() {
	close(l.logCh)
}

func main() {
	var l0 = logger{}
	l0.init()
	defer l0.destroy()

	l0.log(logEntry{time.Now(), logDebug, "hello"})
	l0.log(logEntry{time.Now(), logDebug, "world"})
}
