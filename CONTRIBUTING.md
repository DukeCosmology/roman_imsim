# Guidelines for contributing to the project

- **Create branches**

    The `main` branch is protected and cannot be pushed to directly.
    Create a new branch for your changes, and make sure to name it descriptively.
    If you are working on independent features simultaneously, consider using a feature branch for each one separately.

- **Commit often**

    Make small, incremental commits as you code.
    Ideally, each commit should be atomic and represent a single logical change.
    Committing often helps to keep track of changes and recover any code that you may have deleted.

- **Write clear commit messages**
    Avoid unhelpful commit messages like "Update sca.py".
    Taking the time to write useful commit messages helps identifying them later.
    Multiline commit messages are more helpful than code commits.
    Focus on the 'why' in the message rather than the 'what'.
    Avoid having multiple commits with the same message.

- **Push continuously**

    Do not keep working on your local branch indefinitely.
    Push your changes to the remote repository frequently to ensure that your work is backed up and visible to collaborators.

- **Raise a (draft) pull request**

    Even better than pushing is to raise a pull request (PR) as soon as you have made some progress on your feature.
    This allows others to see your work, provide early feedback, and collaborate more effectively.
    Marking it as a draft indicates that it is not yet ready for a detailed review (or merging), but that you are open to suggestions and improvements.
    This prevents the situation of requiring a major change after spending too much time on a feature.

- **Merge to `main` when done**

    Once your changes are complete and have been reviewed, merge your branch into the `main` branch.
    Code that is not in `main` is not searchable.
    A feature is not considered done until it is merged into `main`.

- **Rebase before merging**

    Avoid merging the `main` branch into your feature branch, as this can create unnecessary merge commits and complicate the commit history.
    Before merging your changes, rebase your branch onto the latest `main` branch to ensure that your changes are up-to-date and to avoid merge conflicts.
    This can be done with the following command:
    ```bash
    git fetch origin
    git rebase origin/main
    ```
- **Delete unused code instead of commenting it out**

    Anything that is committed can always be recovered.
    If you have code that is no longer needed, remove it altogether instead of commenting it out.
    This helps keep the codebase clean and maintainable.
    With `git`, you can recover the code from the commit history if needed.

- **Test your code**

    Test your code thoroughly in addition to the unit tests.
    Write unit tests for the feature that you develop.
