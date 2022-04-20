package main

import "fmt"

func serialize(x chan<- int, a ...int) {
	for _, i := range a {
		x <- i
	}
	close(x)
}

func stage1(a ...int) <-chan int {
	x := make(chan int)
	go serialize(x, a...)
	return x
}

func double(in <-chan int, x chan<- int) {
	for i := range in {
		x <- i * i
	}
	close(x)
}

func stage2(in <-chan int) <-chan int {
	x := make(chan int)
	go double(in, x)
	return x
}

func main() {
	out := stage2(stage1(1, 2, 3, 4, 5))
	for i := range out {
		fmt.Println(i)
	}
}
