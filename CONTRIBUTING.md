# Contributing to Synthetic Data Generation for Smart City Applications

If you are interested in contributing to this project, your contributions will fall
into three categories:
1. You want to report a bug, feature request, or documentation issue
    - Open an issue in this repository describing what you encountered or what you want to see changed.
    - Please include environment details (OS, GPU, driver version, CUDA version, Docker Engine/Compose versions, NVIDIA Container Toolkit),
    reproduction steps, and relevant logs.
    - The maintainers will evaluate and triage issues. If you believe the issue needs priority attention,
    comment on the issue to notify the team.
2. You want to propose a new Feature and implement it
    - Post about your intended feature, and we shall discuss the design and
    implementation.
    - Once we agree that the plan looks good, go ahead and implement it, using
    the [code contributions](#code-contributions) guide below.
3. You want to implement a feature or bug-fix for an outstanding issue
    - Follow the [code contributions](#code-contributions) guide below.
    - If you need more context on a particular issue, please ask and we shall
    provide.

## Code contributions

### Developer Certificate of Origin (DCO)

This project uses the Developer Certificate of Origin for external contributions. By contributing, you agree to the terms at https://developercertificate.org/ and you must sign off your commits.

To sign off a commit, add the following line to your commit message (use `-s` to have git add this automatically):

```
Signed-off-by: Your Name <your.email@example.com>
```

Example workflow:

```
git commit -s -m "feat: add new widget"
```

Please ensure every commit in your PR includes a valid DCO sign-off.

### IP review process for on-going modifications

For on-going project code modifications or contributions by third parties, follow NVIDIA's IP review process: https://nv/ip_review_process

Contact the maintainers if you have questions about applicability.

### Your first issue

1. Read the repository README to learn how to set up the development environment and run the stack.
2. Find an issue to work on. Look for the “good first issue” or “help wanted” labels, or ask for guidance in an issue.
3. Comment on the issue saying you are going to work on it.
4. Get familiar with the repository structure and tooling described in the README and sample notebooks.
5. Code! Make sure to update unit tests!
6. When done, open a Pull Request (PR) against the default branch.
7. Verify that CI passes all status checks, or fix issues if needed.
8. Wait for maintainers to review your code and update code as needed.
9. Once reviewed and approved, a maintainer will merge your pull request.

Remember, if you are unsure about anything, don't hesitate to comment on issues and ask for clarifications!

### Managing PR labels

Each PR should be labeled according to whether it is a “breaking” or “non-breaking” change (using GitHub labels). This helps communicate changes users should know about when upgrading.

For this project, a “breaking” change is one that modifies the public, non-experimental Python API in a non-backward-compatible way. Backward-compatible API changes (e.g., adding a new keyword argument) do not need the breaking label.

Please also add labels indicating whether the change is a feature, improvement, bugfix, or documentation change.

### Seasoned developers

Once you are more comfortable with the code, check open issues and milestones to find higher-impact items.

Look at unassigned issues and comment to claim one. If you have questions related to implementation, ask in the issue rather than only in the PR.

### Branches and Versions

This repository typically has two main branches:

1. `main` branch: contains the latest released or stable version. Only hotfixes are targeted and merged into it.
2. `branch-x.y`: development branch which contains the upcoming release. New features should be based on this branch (with the exception of hotfixes).

### Additional details

For every new version `x.y` there is a corresponding branch called `branch-x.y`, where new feature development starts and PRs are targeted and merged before release. The exceptions to this are 'hotfixes' that target the `main` branch to fix critical issues; when submitting a hotfix, state the intent clearly in the PR.

For all development, push your changes into a branch (created using the naming instructions below) in your own fork of this repository and then open a pull request when the code is ready.

A few days before releasing version `x.y` the code of the current development branch (`branch-x.y`) will be frozen and a new branch, 'branch-x+1.y' will be created to continue development.

### Branch naming

Branches used to create PRs should have a name of the form `<type>-<name>`
which conforms to the following conventions:
- Type:
    - fea - For if the branch is for a new feature(s)
    - enh - For if the branch is an enhancement of an existing feature(s)
    - bug - For if the branch is for fixing a bug(s) or regression(s)
- Name:
    - A name to convey what is being worked on
    - Please use dashes or underscores between words as opposed to spaces.

## Attribution
Portions adopted from https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md
