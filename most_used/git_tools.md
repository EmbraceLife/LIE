# Most Used Git functionalities

### fork, sync the official repo
- make my fork from official
- git clone from my fork
- cd my fork, and `git remote add upstream official-url-git`
- then `git pull upstream master` to sync with official repo
- then `git merge upstream/master`, when necessary
- any changes in local master or other branches, `git push` will update
- use `pull request` on github to push changes to official

### delete a remote branch
- `git push origin --delete a_remote_branch_name`: to delete a branch remote in github

### download the master branch only 
- `git clone --single-branch official_repo_address`: clone a single master branch of the repo

### download purely a folder from remote repo
- `svn checkout url-folder-replace-tree/master-with-trunk`

### Create, rename, delete, switch branches
- `git branch`: check all branches
- `git branch new_branch_name`: create a branch from where we are
- `git branch -m a_new_name`: rename
- `git branch -d branch_to_go`: delete
- `git checkout new_branch`: switch to a new branch

### merge two branches
- `git merge new_branch`: from where we are, merge new_branch into where we are.

### remove uncommitted changes
- `git stash`: put the changes away
- `git stash list`: check the changes being put away
- `git stash apply`: get the changes back
- `git stash drop`: remove the changes for good


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




### add a particular file
```
 git add [some files] # add [some files] to staging area
 git add [some more files] # add [some more files] to staging area
 git commit # commit [some files] and [some more files]
```
