# Kur inside out

## How kur build a computational graph with keras?
- see [commit](https://github.com/EmbraceLife/kur/commit/6e91d0d851b702a08686938623f26dd3bdb40004)

## How kur does the same with pytorch?

## How to I know I am doing good
- get loss plot
- get accuracy plot


## How to debug with pdb in python? 
https://www.youtube.com/watch?v=bZZTeKPRSLQ (simple 8 mins)
https://www.youtube.com/watch?v=P0pIW5tJrRM (in-depth 30 mins)
https://docs.python.org/3/library/pdb.html#debugger-commands official doc for `pdb`
https://pypi.python.org/pypi/pdbpp/ official doc for `pdb++`
- to also use `pdb++`, just install it, then everything is the same as using pdb
- `from pdb.set_trace as set_trace`
- `(pdb): l`: to see the next 11 lines
- `ll`: long list: show the complete function, instead of every 11 lines
- `(pdb): l 20`: to start at line 20 and see the next 11 lines
- `(pdb): l 1, 20`: see line from 1 to 20
- `(pdb): s`: step into a function
- `(pdb): n`: run the next line
- `(pdb): w`: call stack, where I started and where I am in source code, `d` go down a stack, `u` to go up a stack
- `(pdb): b`: see the list of breakpoints we set_trace, and how many times the breakpoint line has been hit
- `b file.py:41` or `b func_name`
- `b 11, this_year==2017`: conditional breakpoint, at line 11 to breakpoint, if this_year == 2017
- `b 20`: set a breakpoint at line 20
- `cl 1`: clear the first breakpoint (when you want to set a new breakpoint, has to clear the previous one first)
- `(pdb): r `: run until the current function returns
- `(pdb): c`: continue to run until the next breakpoint (finished the current block of lines, or finished debugging process, let it run to the end)
- `q`: quit
- `(pdb): ?` or `? list`: for docs
- hit `return`: just repeat the previous command
- `pp variable_name`: nice printing a variable
- autocompletion is available in pdb++
- `sticky`: `n` won't break the continuity of source context code flow
- `sticky` works with `s`: to step into a function too
- `until`: to go to the end of a loop, and put current line there, and also gives you the return value
- `interact`: build upon the current env, and build functions even classes with python codes upon it, and check all variables or expressions
- `pm`: how to use it?
- `a`: print out list of current function

### How to make alias to work everywhere?
- go to home directory: `cd`
- go inside .bash_profile: `nano .bash_profile`
- add a line like `alias wk3.5='cd ~/.Downloads; source activate dlnd-tf-lab' `
- save and exit: `ctrl + x`, `y`, `enter`
- source to activate: `source .bash_profile`
- then use `wk3.5` everywhere

### How to make alias for pdb?
https://youtu.be/lnlZGhnULn4?t=1481
https://github.com/nyergler/in-depth-pdb/blob/master/slides.rst github slides on extension
- check examples inside
- `nano ~/.pdbrc`, however, it seems not working!!!!

### How to do step_watch and next_watch in pdb?
- This adds two commands, nextwatch and stepwatch which each take a variable name varname as an argument. They will make a shallow copy of the current frame's local variable for varname if possible, and keep executing next or step respectively until what that name points to changes.

```bash
alias dr pp dir(%1)
alias dt pp %1.__dict__
alias pdt for k, v in %1.items(): print(k, ": ", v)
alias loc locals().keys()
alias doc from inspect import getdoc; from pprint import pprint; pprint(getdoc(%1))
alias source from inspect import getsourcelines; from pprint import pprint; pprint(getsourcelines(%1))
alias module from inspect import getmodule; from pprint import pprint; pprint(getmodule(%1))
alias fullargs from inspect import getfullargspec; from pprint import pprint; pprint(getfullargspec(%1))

!global __currentframe, __stack; from inspect import currentframe as __currentframe, stack as __stack
!global __copy; from copy import copy as __copy
!global __Pdb; from pdb import Pdb as __Pdb
!global __pdb; __pdb = [__framerec[0].f_locals.get("pdb") or __framerec[0].f_locals.get("self") for __framerec in __stack() if (__framerec[0].f_locals.get("pdb") or __framerec[0].f_locals.get("self")).__class__ == __Pdb][-1]

alias _setup_watchpoint !global __key, __dict, __val; __key = '%1'; __dict = __currentframe().f_locals if __currentframe().f_locals.has_key(__key) else __currentframe().f_globals; __val = __copy(%1)

alias _nextwatch_internal next;; !if __dict[__key] == __val: __pdb.cmdqueue.append("_nextwatch_internal %1")
alias _stepwatch_internal step;; !if __dict[__key] == __val: __pdb.cmdqueue.append("_stepwatch_internal %1")

alias nextwatch __pdb.cmdqueue.extend(["_setup_watchpoint %1", "_nextwatch_internal"])
alias stepwatch __pdb.cmdqueue.extend(["_setup_watchpoint %1", "_stepwatch_internal"])
```
