# Most Used Git functionalities

### switch temporarily to previous commit
- `git log` to get many previous commit numbers
- `git checkout 'commit number'` to go to a previous commit state
- `git branch old_commit_branch` to make a new branch on that previous commit
- `git checkout branch_you_want_to_work` as you are now in a detached head mode

### see differences between previous commit and current head
- after commit, I can still see diff between previous commit and current head with `git show`
- `git diff commit_id HEAD`: see diff between a particular commit and current head
- `git diff`: see diff of current changes before commit it

### change previous commit message
- `git commit --amend -m "new commit message you want to update"`
- if the previous commit is already pushed, then add the following line
- `git push --force`
