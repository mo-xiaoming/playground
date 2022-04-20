package main

import (
	"io"
	"os"
	"strings"
)

// https://gist.github.com/edwardmp/3aca97114eb19089e18d

type rot13Reader struct {
	r io.Reader
}

func (r rot13Reader) Read(b []byte) (int, error) {
	var table = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLM"
	n, err := r.r.Read(b)
	if err == nil {
		for i := 0; i < n; i++ {
			var c = b[i]
			if 'A' <= c && c <= 'Z' {
				b[i] = table[c-'A'+13]
			} else if 'a' <= c && c <= 'z' {
				b[i] = table[c-'a'+39]
			}
		}
	}
	return n, err
}

func main() {
	s := strings.NewReader("Lbh penpxrq gur pbqr!")
	r := rot13Reader{s}
	io.Copy(os.Stdout, &r)
}
