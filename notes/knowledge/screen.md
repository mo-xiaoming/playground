```bash
screen -S session_name  # create a named session

screen -L -S session_name   # log everything

screen -r [session_name]    # re-attach to a session

screen -ls              # list sessions

screen -d -R session_name   # detach then re-attach to a session
```

```
^a ?    # help

^a c    # create a new window

^a "    # list all windows

^a 0    # switch to window #

^a n    # switch to next window

^a p    # switch to previous window

^a A    # rename current window

^a S    # split current region horizontally

^a |    # split current region vertically

^a TAB  # switch focus between regions

^a ^a   # toggle between current and previous regions

^a Q    # close all region but current one

^a X    # close current region

^a [    # copy mode

^a ]    # paste

^a d    # detach

^a H    # log

^a M    # monitor for activity

^a _    # monitor for silence

^a x    # lock screen

^a k    # kill session

```

in copy mode, `c` to set left margin, `C` to set right margin
