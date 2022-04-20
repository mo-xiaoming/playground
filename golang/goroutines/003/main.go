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

type logger interface {
	log(entry logEntry)
}

type defaultLogger struct {
}

func (l *defaultLogger) log(entry logEntry) {
	fmt.Printf("%v [%s] %s\n", entry.when.Format("2006-01-02T15:04:05"), entry.level, entry.msg)
}

func main() {
	var l0 = defaultLogger{}
	l0.log(logEntry{time.Now(), logDebug, "hello"})
	l0.log(logEntry{time.Now(), logDebug, "world"})
}
