```
E1 = E2
```

*left* operand `E1` must be an lvalue, an lvalue is an expression refer to an object, an object is a region of storage

```cpp
int n;
n = 1;
```

`n` is an lvalue object. `1` is an rvalue, anything isn't an lvalue is a rvalue

```cpp
x[n + 1] = abs(p->value);
```

`x[n+1]` has to be an lvalue, it must refer to an object, the right side can be either an lvalue or rvalue

Why make this distiction between these two

Rvalues don't necessarily occupy storage, that offers freedom in code generation

`1 = n` is an error, because `1` doesn't occupy a memory storage

* lvalue is an expression refers to an object
* rvalue is any expression that is not an lvalue

Most literals are rvalues, they don't necessarily occupy any storage

However, string literals "xyz" do occupy storage, it is an array, they are lvalue

the return value of an operation often goes to a register, such value are rvalues

`m + 1 = n`, it is an error because `m+1` is an rvalue

\* yields an lvalue
\& yields an rvalue

```cpp
int i = 0;
int j = 0;

int &r = i;
&r = j; // Error
```

Conceptually, rvalues (of non-class type) don't occupy data storage, some might. C/C++ program insist that you program as if non-class rvalue doesn't occupy space

Conceptually, lvaues (of any type) occupy data storage, in truth, compiler might eliminate some of them. C/C++ let you assume that lvalue always occupy storage

Conceptually, rvalues of class type do occupy space

```cpp
struct S { int x, y; }

S foo();

int n = foo().y;
```

The return value of `foo` must have a base address, thus it has to occupy memory space, that's why rvalue of class type must be treated differently

Not all lvalues can appear on the left of assignment, `const`

|                       | & | = |
|-----------------------|---|---|
| lvalue                | Y | Y |
| non-modifiable lvalue | Y | N |
| non-class rvalue      | N | N |

references are pointers can automatically deference, for overload operators behave like built-in types

`f(T &t)` can only accept modifiable lvalue,

```cpp
int *p = &3; // Error
int &r = 3;  // Error
```

Both are rvalues

reference to const T, can bind to an expression x that's not an lvalue of type T, or there is an conversion from x to T. in that case, compiler creates an tempory object to hold a copy of x converted to T, this makes reference have something to bind to, that tempory will be destroyed when it goes out of scope. in the latter case, passing argument by reference can be more expensive than passing by value

The temporary object created is an rvalue but occupies memory for both built-in and class type

*pure rvalue* or prvalue, doesn't occupy space
*expiring rvalue* or xvalue, which does

a temporary object is created via *temporary materialization conversion*, it converts a prvalue to xvalue

C++11 introduces *lvalue reference* = pre-C++11 *value reference*, *rvalue reference* uses `&&`, which only binds rvalues

Binding rvalue reference to an rvalue triggers a temporary materialization conversion, just like binding a lvalue reference to const to an lvalue

Mordern C++ use rvalue references to avoid unnecessary copying

```cpp
std::string s1, s2, s3;

s1 = s2;	// copy assignment

s1 = s2 + s3; // move assignment
```

`std::move` casts lvalue to an xvalue

glvalue : generalized lvalue
xvalue : an 'expiring' lvaue
prvalue: 'pure' rvalue
