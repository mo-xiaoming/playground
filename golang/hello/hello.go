package main

import (
	"fmt"
	"github.com/google/go-cmp/cmp"
	"skelix.net/moxiaoming/hello/morestrings"
)

func main() {
	fmt.Println(morestrings.ReverseRune("!oG ,olleH"))
	fmt.Println(cmp.Diff("Hello World", "Hello Go"))
}
