## reusing binding values

```cpp
std::boyer_moore_searacher bmsearch{sub.begin(), sub.end()};
for (auto [beg, end] = bmsearch(text.begin(), text.end());
     beg != text.end();
     std::tie(beg, end) = bmsearch(end, text.end())) {
}
```

## structure binding interface

`tuple_size`, `tuple_element` and `get`

[seems less efficient](https://godbolt.org/z/c6WzG8vx6)

## lock guard

```cpp
if (auto lg = std::lock_guard(mutex); !col1.empty()) {
    std::cout << col1.front() << '\n';
}
```
