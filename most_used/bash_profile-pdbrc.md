
### How to make alias in `.bash_profile`?
- go to home directory: `cd`
- go inside .bash_profile: `nano .bash_profile`
- add a line like `alias wk3.5='cd ~/.Downloads; source activate dlnd-tf-lab' `
- save and exit: `ctrl + x`, `y`, `enter`
- source to activate: `source .bash_profile`
- then use `wk3.5` everywhere

### How to make alias for pdb?
- go to home directory, save the following code inside: `nano ~/.pdbrc`
- Important: No `source ~./pdbrc` at all!

```bash
alias dr pp dir(%1)
alias dt pp %1.__dict__
alias pdt for k, v in %1.items(): print(k, ": ", v)
alias loc locals().keys()
alias doc from inspect import getdoc; from pprint import pprint; pprint(getdoc(%1))
alias sources from inspect import getsourcelines; from pprint import pprint; pprint(getsourcelines(%1))
alias module from inspect import getmodule; from pprint import pprint; pprint(getmodule(%1))
alias fullargs from inspect import getfullargspec; from pprint import pprint; pprint(getfullargspec(%1))
alias opt_param optimizer.param_groups[0]['params'][%1]
alias opt_grad optimizer.param_groups[0]['params'][%1].grad
```
