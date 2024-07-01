# Contributing to `Case_Studies_L2D`  
👍🎉 First off, thanks for taking the time to contribute! 🎉👍

### How to Contribute

The easiest way to get started is to file an issue to tell us about a spelling
mistake, some awkward wording, or a factual error. This is a good way to
introduce yourself and to meet some of our community members.

1. If you have a [GitHub][github] account, or are willing to [create
   one][github-join], but do not know how to use Git, 
   you can open an issue (bug report, feature request, or something is not working)
   https://github.com/LearnToDiscover/Case_Studies_L2D/issues/new/choose
   This allows us to assign the item to someone and to respond to it in a threaded discussion.

2. If you are comfortable with Git, and would like to add or change material,
   you can submit a pull request (PR). Instructions for doing this are
   [included below](#using-github).


### :octocat: Using GitHub

#### Setting up your repository locally. 
1. Generate your SSH keys as suggested [here](https://docs.github.com/en/github/authenticating-to-github/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)
2. Clone the repository by typing (or copying) the following lines in a terminal
```
git clone git@github.com:LearnToDiscover/Case_Studies_L2D.git
```

#### Committing and pushing changes 
1. Create new branch using issue number
```
git checkout -b ISSUENUMBER-branch-name 
```
2. Commit changes and push to your branch
```
git add .
git commit -m 'short message (#ISSUENUMBER)'
git push origin ISSUENUMBER-branch-name
```
3. Submit a Pull Request against the `main` branch. 

#### Pull Request (PR) and merge to `main` branch
1. Select branch that contain your commits.
2. Click `Compare and pull request` and create PR for the associated branch.
3. Type a title and description of your PR and create PR
4. Please keep your PR in sync with the base branch.
```
git checkout main
git pull origin main
git checkout FEATURE_BRANCH
git rebase main
git push --force origin FEATURE_BRANCH
```
4.1 In case you are in a different `MY_FEATURE_BRANCH` branch, follow:
```
git checkout FEATURE_BRANCH
git pull origin FEATURE_BRANCH
git checkout MY_FEATURE_BRANCH 
git rebase FEATURE_BRANCH
git push --force origin MY_FEATURE_BRANCH
```
5. Run `pre-commit` to tidy up code and documentation (see next section). 
6. Request a PR review.
See [collaborating-with-pull-requests](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests) for further details.
7. Once your PRs has been approved, procced to merge it to main. See [Merging a pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/incorporating-changes-from-a-pull-request/merging-a-pull-request)
8. Remove your merged branch from your repo and in the list of https://github.com/LearnToDiscover/Case_Studies_L2D/branches
```
#Local git clear
git branch --merged | grep -v '\*\|master\|main\|develop' | xargs -n 1 git branch -d
#Remote git clear
git branch -r --merged | grep -v '\*\|master\|main\|develop' | sed 's/origin\///' | xargs -n 1 git push --delete origin
```

NB: The published copy of the lesson is usually in the `main` branch.

Each lesson has a team of maintainers who review issues and pull requests or
encourage others to do so. The maintainers are community volunteers, and have
final say over what gets merged into the lesson.

[repo]: https://example.com/FIXME
[dc-issues]: https://github.com/issues?q=user%3Adatacarpentry
[github]: https://github.com
[github-flow]: https://guides.github.com/introduction/flow/
[github-join]: https://github.com/join
[how-contribute]: https://egghead.io/series/how-to-contribute-to-an-open-source-project-on-github
[lc-issues]: https://github.com/issues?q=user%3ALibraryCarpentry
[swc-issues]: https://github.com/issues?q=user%3Aswcarpentry
