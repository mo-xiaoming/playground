The *source language" is a set, usually _infinite_, of strings defined by some _finite set of rules_, called a *grammar*

A scanner takes a stream of characters and converts it to a stream of classified words - that is, pairs fo the form (p, s), where _p_ is the word's part of _speech_ and _s_ is its spelling.

A scanner would convert the string into the following stream of classified words:
  (noun, "Compilers"), (verb, "are"), (adjective, "engineered"), (noun, "object"), (endmark, ".")

The process of automatically finding derivations is called *parsing*, the compiler pass that determines if the input stream is a *sentence* in the source language

A grammatically correct sentence can be meaningless. for example, "Rocks are green vegetables"

*Regular language* can be recognized using *finite automaton* and *regular expressions*, and can also use *regular grammar*

NFA vs DFA,
  - NFA can have multiple transitions for the same input symbol from one state
  - NFA can some transitions for some input symbols from one state
  - NFA can accept epsilon

```
               human language       programming language
scanning ----> words             -> tokens
parsing  ----> sentences         -> statements
```

The language described by an RE is called a *regular language*

- Σ: characters contained in some alphabet + a character ε that represents the empty string
- RE: a set of strings over Σ, _r_
- language: the set of strings, _L(r)_

_RE_ has three basic operations:
1. Alternation/Union
2. Concatenation
3. Closure/Star

Finite Closure: _R<sup>i</sup>_ designates 1 to _i_ occurrences of _R_
Positive Closure: _R<sup>+</sup>_ denotes 1 or more occurrences of _R_

*identifiers*: `([A...Z]|[a...z])([A...Z]|[a...z]|[0...9]|_)*`
*unsigned integer*: `0|[1...9][0...9]*`
*unsigned real numbers*: `(0|[1...9][0...9]*)(ε|.[0...9]*)`E(ε|+|-)(0|[1...9][0...9]*)`
*quoted character strings*: `"(^")*"
*line comments*: `//(^\n)*\n`
*block comments*: `/*(^*|*+^/)**/`




A *formal language* consists of a sequence of valid *tokens*
From scanning point of view, a valid *program* that consists of a sequence of valid *tokens*
And each *token* belongs to a valid part of speech.
The part of *speech* here is *identifier*

