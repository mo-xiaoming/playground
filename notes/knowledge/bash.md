for bash scripts
```base
set -e # fail on error
set -u # fail on unset variable
set -o pipeline # fail during pipeline

set -euo pipeline # on every bash script
```

`^X ^E` launch an editor ($EDITOR or nano)

`^X Backspace` removes all text from the cursor to the beginning of the line

`^T` transpose the character before the cursor with the character under the cursor

`ALT T` transposes the two words immediately before (or under) the cursor

`ALT C` upper case the letter under the cursor

`!!` execute last command

`!top` execute the most recent command starts with `top`

`!top:p` display the command that `!top` would run (also adds it to the latest command in the command history)

`!$`/`ALT .` execute the last word of the previous command

`!$:p` displays the word that `!$` would execute

`!*` display the last world of the previous command

`!*:p` displays the word that `!*` would execute

`!^` 2nd word of previous command


`cat -n file1 - file2` concatenate files with stdin in between

* `-A` print `$` at the end of each ling
*
* `-n` numbers all lines
*
* `-b` numbers lines that are not blank
*
* `-s` reduces a series of blank lines to a single blank line

`cut` by characters `-c`

* `-c 3` the 3rd character
*
* `-c 3-5` from the 3rd to 5rd character
*
* `-c -5` from the 1st to the 5th character
*
* `-c 5-` from the 5th character to the end of the line
*
* `-c 3,5-7` the 3rd and from the 5th to the 7th character
*
* `-d` and `-f` cutting by filed

`fold -w 30 -s longline.txt` make long lines easier to read and breaks at spaces

`tr`

* `A-Z` all uppercase letters
* `a-z0-9` lowercase letters and digits
* `\n[:punct:]` newline and punctuation characters

* `tr SET1 SET2` translates one set to another
*
* `tr -d SET` delete a set of characters
*
* `tr -dc SET` delete the complement of a set of characters

```bash
while ! ./run; do
	sleep 10
done
```

```bash
until ./run; do sleep 10; done
```

find exclude folder
```bash
find . -type f -not -path "./dir1/*" -not -path "./dir4/*" -exec cp '{}' /tmp \;
```
