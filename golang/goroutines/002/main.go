package main

import (
	"fmt"
	"sync"
)

var wg = sync.WaitGroup{}


func main() {
	var ch = make(chan int, 2)

	wg.Add(2)
	go func(c <-chan int) {
		if i, ok := <-ch; ok {
			fmt.Println(i)
		}
		i := <-c
		fmt.Println(i)
		for i := range ch {
			fmt.Println(i)
		}
		wg.Done()
	}(ch)
	go func(c chan<- int) {
		c <- 42
		c <- 17
		close(c)
		wg.Done()
	}(ch)

	wg.Wait()
}
