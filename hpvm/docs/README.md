# Building docs

We use Sphinx for generating the API and reference documentation.

## Instructions

Install Sphinx needed to build the documentation by entering:

```bash
pip3 install -r requirements.txt
```

To build the HTML documentation, enter::

```bash
make html
```

in the ``doc/`` directory. This will generate a ``build/html`` subdirectory
containing the built documentation.

To build the PDF documentation, enter::

```bash
make latexpdf
```

You will need to have LaTeX installed for this.
