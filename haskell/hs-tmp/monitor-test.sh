#/usr/bin/env bash

function ticks {
  echo $(date '+%s')
}

function eqTicks {
  local first=$1
  local second=$2
  echo $(( (first + 5) >= second ))
}

lastUpdate=$(ticks)

inotifywait @./.git* @./dist-newstyle -q -r -e modify,create,delete -m . | \
  while read path event file; do
    [ $(eqTicks $lastUpdate $(ticks)) == 1 ] && continue
    lastUpdate=$(ticks)

    tput clear
    if [ -f $path$file ]; then
      echo -e '===============' $path$file $event '================='
      cabal test
    fi
  done
