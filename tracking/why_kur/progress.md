# Kur inside out

## How kur build a computational graph with keras?
- see [commit](https://github.com/EmbraceLife/kur/commit/6e91d0d851b702a08686938623f26dd3bdb40004)

## How kur does the same with pytorch?

## How to I know I am doing good
- get loss plot
- get accuracy plot


## How to debug kur?
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
- `interact`: build upon the current env, and build functions even classes with python codes upon it

### When to use pdb?
https://www.youtube.com/watch?v=lnlZGhnULn4
- 
