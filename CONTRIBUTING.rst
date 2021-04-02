.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/vintasoftware/entity-embed/issues

Before reporting a bug, please double-check the requirements of Entity Embed: https://github.com/vintasoftware/entity-embed/blob/main/README.md#requirements

If you think you really found a bug, please create a GitHub issue and use the "Bug report" template.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help wanted" is open to whoever wants to implement it. Please comment on the issue saying you're working in a solution.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement" and "help wanted" is open to whoever wants to implement it. Please comment on the issue saying you're working in a solution.

Write Documentation
~~~~~~~~~~~~~~~~~~~

entity-embed could always use more documentation, whether as part of the official entity-embed docs, in docstrings, or even on the web in blog posts, articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

If you have a suggestion, concern, or want to propose a feature, please create a GitHub issue and use the "New feature" template.

Get Started!
------------

Ready to contribute? Please read our Code of Conduct: https://github.com/vintasoftware/entity-embed/blob/main/CODE_OF_CONDUCT.md

Now, here's how to set up `entity-embed` for local development.

1. Fork the `entity-embed` repo on GitHub.
2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/entity-embed.git

3. Install your local copy into a virtualenv. Assuming you have virtualenvwrapper installed, this is how you set up your fork for local development::

    $ mkvirtualenv entity-embed
    $ cd entity-embed/
    $ python setup.py develop

4. Install dev requirements::

    $ pip install -r requirements-dev.txt

5. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

6. When you're done making changes, check that your changes pass flake8 and the
   tests, including testing other Python versions with tox::

    $ flake8 entity-embed tests
    $ pytest
    $ tox

   To get flake8 and tox, just pip install them into your virtualenv.

7. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

8. Submit a Pull Request through the GitHub website.

Pull Request Guidelines
-----------------------

Before you submit a Pull Request, check that it meets these guidelines:

1. The Pull Request should include tests.
2. If the Pull Request adds functionality, the docs should be updated.
3. The CI should pass.
