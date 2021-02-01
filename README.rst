.. raw:: html

    <embed>
        <p align="center">
            <img width="300" src="https://github.com/yngtodd/molecules/blob/master/img/molecules.png">
        </p>
    </embed>

--------------------------

.. image:: https://badge.fury.io/py/molecules.png
    :target: http://badge.fury.io/py/molecules
    
.. highlight:: shell

=========
Molecules
=========

Machine learning for molecular dynamics.

Documentation
--------------

For references, tutorials, and examples check out our `documentation`_.

Installation
------------
To install a conda environment:

.. code-block:: console

    git clone https://github.com/braceal/molecules.git
    conda env create -f env.yaml -p ./conda-env

For development install via pip:

.. code-block:: console

    git clone https://github.com/braceal/molecules.git
    python3 -m venv env
    source env/bin/activate
    pip install --upgrade pip setuptools wheel
    pip install -e .

Then, install pre-commit hooks: this will auto-format and auto-lint on commit to enforce consistent code style:

.. code-block:: console

    pre-commit install
    pre-commit autoupdate

.. _documentation: https://molecules.readthedocs.io/en/latest
