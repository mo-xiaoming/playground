# Common idiom

## Result: return Err to upper, assign when Ok

```rust
let i = match h() {
  Ok(i) = i,
  err = return err,
}

///////// to

let i = h()?;
```

## Option: map when has Some, otherwise None

`Option<T> -> Option<U>`

```rust
fn add_four(x: i32) -> i32 {
  x + 4
}

fn maybe_add_four(y: Option<i32>) -> Option<i32> {
  match y {
    Some(yy) => Some(add_four(yy)),
    None => None
  }
}


///////// to

fn maybe_add_four(y: Option<i32>) -> Option<i32> {
  y.map(add_four)
}
```

## Option: do something when Some, otherwise still None

`Option<T> -> Option<U>`

```rust
fn foo(input: Option<i32>) -> Option<i32> {
  let input = input?;
  if input < 0 {
    return None;
  }
  Some(input)
}

///////// to

fn foo(input: Option<i32>) -> Option<i32> {
  input.and_then(|i| {
    if i < 0 {
      None
    } else {
      Some(i)
    }
  }
}

///////// to

fn foo(input: Option<i32>) -> Option<i32> {
  input.filter(|i| *i > 0)
}
```

## default for Option

can be used for `Option<T> -> Result<T, Error>`

```rust
fn foo(input: Option<i32>) -> Result<i32, ErrNegative> {
  match input {
    Some(n) => Ok(n),
    None => ErrNegative,
  }
}

///////// to

fn foo(input: Option<i32>) -> Result<i32, ErrNegative> {
  input.ok_or(ErrNegative)
}
```

## iterator instead of for

```rust
fn ping_all(foos: &[Foo]) {
  for f in foos {
    f.ping();
  }
}

///////// to

fn ping_all(foos: &[Foo]) {
  foos.iter().for_each(|f| f.ping());
}
```

## map, filter, for_each, chain, enumerate

```rust
let vec = vec![0, 1, 2, 3];
vec.iter()
   .map(|x| x + 1)
   .filter(|x| x > 1)
   .for_each(|x| println!("{}" x));

for (i, v) in vec.iter()
                 .chain(vec![4, 5, 6, 7].iter())
                 .enumerate() {
  // do something
}
```

## collect

```rust
let vec = vec![0, 1, 2, 3];
let vec_2: Vec<_> = vec.iter().map(|x| x + 2).collect();
let map: HashMap<_, _> = vec.iter().map(|x| x * 2).enumerate().collect();
```

## add Option to a Container

```rust
let grade = Some("A+");
let mut grades = vec!["B-", "C+", "D"];

if let Some(grade) = grade {
  grades.push(grade);
}

///////// to

// 1. extend accept an iterator
// 2. Option implements into iterator
grades.extend(grade);

// without allocation
for grade in grades.iter().chain(grade.iter()) {
  println!("{grade}");
}
```

## filter out None in a vector of Option

```rust
let grades = vec![Some("A"), None, Some("B-"), None];
// flatten extracts the value from Some, and discard None
let grades: Vec<&str> = grades.into_iter().flatten().collect();
println!("{grades:?}");
```

## filter_map

```rust
let grades = vec!["3.8", "B+", "4.0", "A", "2.7"];
let grades: Vec<f32> = grades.iter()
                             .map(|s| s.parse())
                             .filter(|s| s.is_ok())
                             .map(|s| s.unwrap())
                             .collect();

///////// to

// ok convert Result to Option, and filter_map only extracts the result from Some
let grades: Vec<f32> = grades.iter()
                             .filter_map(|s| s.parse.ok())
                             .collect();
```

```rust
[derive(Debug)]
struct Student {
    name: String,
    pga: f32,
}

let students = vec![
    "Bogdan 3.1",
    "Wallace 2.3",
    "Lidiya 3.5",
    "Kyle 3.9",
    "Anatoliy 4.0",
];

let good_students: Vec<Student> = students
    .iter()
    .map(|s| {
        let mut s = s.split(' ');
        let name = s.next()?.to_owned();
        let pga = s.next()?.parse::<f32>().ok()?;

        Some(Student { name, pga })
    })
    .flatten()
    .filter(|s| s.pga >= 3.5)
    .collect();

///////// to

let good_students: Vec<Student> = students
    .iter()
    .filter_map(|s| {
        let mut s = s.split(' ');
        let name = s.next()?.to_owned();
        let pga = s.next()?.parse::<f32>().ok()?;

        Some(Student { name, pga })
    })
    .filter(|s| s.pga >= 3.5)
    .collect();

for s in good_students {
    println!("{s:?}");
}
```

## don't check split length

```rust
let parts: Vec<&str> = input.split_whitespace().collect();

if parts.len() != 2 {
  return Err(/*...*/);
}

let part0 = parts[0];
let part1 = parts[1];

////////// to

match parts[..] {
  (part0, part) => /*..*/,
  _ => Err(/*..*/),
}
```

## Type parsed from a string

```rust
impl std::str::FromStr for Money {
  type Err = MoneyError;

  fn from_str(s: &str) -> Result<Self, Self::Err> {
    /*...*/
  }
}

let m = "101.3 euro"::parse::<Money>();
```

## upset a map

```rust
enum Value<'a> {
  Single(&'a str),
  Multiple(Vec<&'a str>),
}

let mut data = HashMap::new();

data.entry(key).and_modify(|existing: &mut Value| match existing {
  Value::Single(prev_val) => {
    *existing = Value::Multiple(vec![val, prev_val]);
  },
  Value::Multiple(vec) => vec.push(val),
}).or_insert(Value::Single(val));
```
