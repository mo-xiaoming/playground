package main

import (
	"context"
	"fmt"
	"github.com/go-redis/redis/v8"
)

func main() {
	opt, err := redis.ParseURL("redis://localhost:6379/0")
	if err != nil {
		panic(err)
	}

	rdb := redis.NewClient(opt)
	var ctx = context.Background()
	pong, err := rdb.Ping(ctx).Result()
	fmt.Println(pong, err)
}
