Rust Programming Techniques-vqavdUGKeb4.mp4
Matthias Endler - Idiomatic Rust-P2mooqNMxMs.mp4
Pascal Hertleif - Writing Idiomatic Libraries in Rust-0zOg8_B71gE.mp4
  - put doctest setup code in a file then `# include!("src/doctest_helper.rs");`
  - layout
     ```
     src/lib.rs
     src/main.rs
     src/bin/{name}.rs
     tests/
     examples/
     benches/
     ```
  - `#![deny(missing_docs)]`
  - implementing traits make it easier to work with, like `let x: IpAddress = [127, 0, 0, 1].into();`
  - `std::convert` is your friend
    - AsRef: reference to reference conversion
    - From/Into: Value conversions
    - TryFrom/TryInto: Falliable conversions
  - implement **ALL** the traits
    - Debug, (Partial)Ord, (Partial)Eq, Hash
    - Display, Error
    - Default
    - (Serde's Serialize + Deserialize)
  - parse string with FromStr, like `"green".parse::<Color>()`
      ```rust
      impl FromStr for Color {
        type Err = UnknownColorError;
        fn from_str(s: &str) -> Result<Self, Self::Error> {}
      }
      ```
  - builder pattern, headers cannot be written after body is set
      ```rust
      HttpResponse::new()      // NewResponse
        .header("Foo", "1")    // WritingHeaders
        .header("Bar", "2")    // WritingHeaders
        .body("Lorem ipsum")   // WritingBody
        .header("Baz", "3")
       //^ Error: no method `header` found for type `WrittingBody`
      ```
  - iterator as input, abstruct over collections, avoid allocations
      ```rust
      fn foo(data: &HashMap<i32, i32>) {}

      // v.s.

      fn bar<D>(data: D) where D: IntoIterator<Item=(i32, i32)> {}
      ```
[E] 'Type-Driven API Design in Rust' by Will Crichton-bnnacleqg6k.mp4
  - progress. iterator, extend all iter(), and bound/unbound change types on the fly: builder pattern
