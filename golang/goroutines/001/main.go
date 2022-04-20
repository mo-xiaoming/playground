package main

import (
	"fmt"
	"sync"
)

var wg = sync.WaitGroup{}

func main() {
	var msg = "this"
	wg.Add(1)
	go func(msg string) {
		fmt.Println(msg)
		wg.Done()
	}(msg)
	msg = "not this"
	wg.Wait()
	// no need to sleep
}
