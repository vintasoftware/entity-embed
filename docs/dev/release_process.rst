Release Process
===============

For maintainers only:

- Update ``CHANGELOG.md`` with the version changes
- Update ``docs/`` as required
- Update ``README.md`` as required
    - Update "Citations" section on ``README.md``
- Commit changes
- Run ``bump2version <minor|major|patch>`` to update the version number (pick one of the options)

    - Version number on ``entity_embed/__init__.py`` and ``setup.py`` will be updated automatically
    - You can specify the ``--new_version`` flag in case you wish to manually set the newest version (if not provided, it will be done automatically based on the chosen option)

- Run ``etc/release.sh`` to generate and upload the new version artifacts
