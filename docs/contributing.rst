=====================
Contribution Guide
=====================

Contributions are highly welcomed and appreciated.  Every little help counts,
so do not hesitate! You can make a high impact on ``ocetrac`` just by using it, being
involved in `discussions <https://github.com/ocetrac/ocetrac/discussions>`_ and reporting `issues <https://github.com/ocetrac/ocetrac/issues>`__.

The following sections cover some general guidelines
regarding development in ``ocetrac`` for maintainers and contributors.

Nothing here is set in stone and can't be changed.
Feel free to suggest improvements or changes in the workflow.


.. contents:: Contribution links
   :depth: 2



.. _submitfeedback:

Feature requests and feedback
-----------------------------

We are eager to hear about your requests for new features and any suggestions about the
API, infrastructure, and so on. Feel free to start a discussion about these on the
`discussions tab <https://github.com/ocetrac/ocetrac/discussions>`_ on github
under the "ideas" section.

After discussion with a few community members, and agreement that the feature should be added and who will work on it,
a new issue should be opened. In the issue, please make sure to explain in detail how the feature should work and keep
the scope as narrow as possible. This will make it easier to implement in small PRs.


.. _reportbugs:

Report bugs
-----------

Report bugs for ``ocetrac`` in the `issue tracker <https://github.com/ocetrac/ocetrac/issues>`_
with the label "bug".

If you can write a demonstration test that currently fails but should pass
that is a very useful commit to make as well, even if you cannot fix the bug itself.


.. _fixbugs:

Fix bugs
--------

Look through the `GitHub issues for bugs <https://github.com/ocetrac/ocetrac/labels/bug>`_.

Talk to developers to find out how you can fix specific bugs.



Preparing Pull Requests
-----------------------

#. Fork the
   `ocetrac GitHub repository <https://github.com/ocetrac/ocetrac>`__.  It's
   fine to use ``ocetrac`` as your fork repository name because it will live
   under your username.

#. Clone your fork locally using `git <https://git-scm.com/>`_, connect your repository
   to the upstream (main project), and create a branch::

    $ git clone git@github.com:YOUR_GITHUB_USERNAME/ocetrac.git # clone to local machine
    $ cd ocetrac
    $ git remote add upstream git@github.com:ocetrac/ocetrac.git # connect to upstream remote

    # now, to fix a bug or add feature create your own branch off "main":

    $ git checkout -b your-bugfix-feature-branch-name main # Create a new branch where you will make changes

   If you need some help with Git, follow this quick start
   guide: https://git.wiki.kernel.org/index.php/QuickStart

#. Set up a [conda](environment) with all necessary dependencies::

    $ conda env create -f ci/environment-py3.8.yml

#. Activate your environment::

   $ conda activate test_env_ocetrac
   *Make sure you are in this environment when working on changes in the future too.*

#. Install the Ocetrac package::

   $ pip install -e . --no-deps

#. Before you modify anything, ensure that the setup works by executing all tests::

   $ pytest

   You want to see an output indicating no failures, like this::

   $ ========================== n passed, j warnings in 17.07s ===========================


#. Install `pre-commit <https://pre-commit.com>`_ and its hook on the ``ocetrac`` repo::

     $ pip install --user pre-commit
     $ pre-commit install

   Afterwards ``pre-commit`` will run whenever you commit. If some errors are reported by pre-commit
   you should format the code by running::

     $ pre-commit run --all-files

   and then try to commit again.

   https://pre-commit.com/ is a framework for managing and maintaining multi-language pre-commit
   hooks to ensure code-style and code formatting is consistent.

    You can now edit your local working copy and run/add tests as necessary. Please follow
    PEP-8 for naming. When committing, ``pre-commit`` will modify the files as needed, or
    will generally be quite clear about what you need to do to pass the commit test.


#. Break your edits up into reasonably sized commits::

    $ git commit -a -m "<commit message>"
    $ git push -u

   Committing will run the pre-commit hooks (isort, black and flake8).
   Pushing will run the pre-push hooks (pytest and coverage)

   We highly recommend using test driven development, but our coverage requirement is
   low at the moment due to lack of tests. If you are able to write tests, please
   stick to `xarray <http://xarray.pydata.org/en/stable/contributing.html>`_'s
   testing recommendations.


#. Add yourself to the `Project Contributors <https://ocetrac.readthedocs.io/en/latest/authors.html>`_ list via ``./docs/authors.md``.


#. Finally, submit a pull request (PR) through the GitHub website using this data::

    head-fork: YOUR_GITHUB_USERNAME/ocetrac
    compare: your-branch-name

    base-fork: ocetrac/ocetrac
    base: main

   The merged pull request will undergo the same testing that your local branch
   had to pass when pushing.


#. After your pull request is merged into the `ocetrac/main`, you will need
   to fetch those changes and rebase your main so that your main reflects the latest
   version of ocetrac. The changes should be fetched and incorporated (rebase) also right
   before you are planning to introduce changes.::

     $ git checkout main # switch back to main branch
     $ git fetch upstream  # Download all changes from central upstream repo
     $ git rebase upstream/main  # Apply the changes that have been made to central repo,
     $ # since your last fetch, onto your main.
     $ git branch -d your-bugfix-feature-branch-name  # to delete the branch after PR is approved
