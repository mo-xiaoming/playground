`head` and `tail`

```haskell
head [5,4,3,2,1]

5

tail [5,4,3,2,1]

[4,3,2,1]
```

`init` and `last`

```haskell
init [5,4,3,2,1]

[5,4,3,2]

last [5,4,3,2,1]

1
```

![head_tail_last_init](http://s3.amazonaws.com/lyah/listmonster.png)

`length` takes a list and returns its length, obviously.

```haskell
length [5,4,3,2,1]

5
```

`null` checks if a list is empty

```haskell
null [1,2,3]

False

null []

True
```

`reverse` reverses a list.

```haskell
reverse [5,4,3,2,1]

[1,2,3,4,5]
```

`take` and `drop`

```haskell
take 3 [5,4,3,2,1]

[5,4,3]

take 1 [3,9,3]

[3]

take 5 [1,2]

[1,2]

take 0 [6,6,6]

[]

drop 3 [8,4,2,1,5,6]

[1,5,6]

drop 0 [1,2,3,4]

[1,2,3,4]

drop 100 [1,2,3,4]

[]
```

`maximum` and  `minimum`

```haskell
minimum [8,4,2,1,5,6]

1

maximum [1,9,2,3,4]

9
```

`sum` and `product`

```haskell
sum [5,2,1,6,3,2,5,7]

31

product [6,2,1,2]

24

product [1,2,5,6,7,9,2,0]

0
```

`elem` takes a thing and a list of things and tells us if that thing is an element of the list.

```haskell
elem 4 [3,4,5,6]

True

elem 10 [3,4,5,6]

False
```

`cycle` takes a list and cycles it into an infinite list. `repeat` like cycling a list with only one element. `replicate` might be a simplier choice

```haskell
take 10 (cycle [1,2,3])

[1,2,3,1,2,3,1,2,3,1]

take 12 (cycle "LOL ")

"LOL LOL LOL "

take 10 (repeat 5)

[5,5,5,5,5,5,5,5,5,5]

replicate 10 5

[5,5,5,5,5,5,5,5,5,5]
```

`fst` and `snd` take a pair, return its 1st and 2nd value correspondingly

```haskell
fst (8, 11)

8

snd (8, 11)

11
```

`zip`

```haskell
zip [5,3,2,6,2,7,2,5,4,6,6] ["im","a","turtle"]

[(5,"im"),(3,"a"),(2,"turtle")]

zip [1..] ["apple", "orange", "cherry", "mango"]

[(1,"apple"),(2,"orange"),(3,"cherry"),(4,"mango")]
```
