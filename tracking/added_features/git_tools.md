# Most Used Git functionalities

### switch temporarily to previous commit
- `git log` to get many previous commit numbers
- `git checkout 'commit number'` to go to a previous commit state
- `git branch old_commit_branch` to make a new branch on that previous commit
- `git checkout current commit number` to go back to current head or commit

### see differences between previous commit and current head
- after commit, I can still see diff between previous commit and current head with `git show`
- `git diff commit_id HEAD`: see diff between a particular commit and current head
- `git diff`: see diff of current changes before commit it 
