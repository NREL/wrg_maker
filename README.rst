#########################
Welcome to the wrg_maker!
#########################

`wrg-maker` converts NREL's HDF5-based wind resource datasets (WINDToolkit) into industry standard WRG format

Installing wrg-maker
====================

Option 1: Install from PIP (recommended for users):
---------------------------
1. Create a new environment:
    ``conda create --name wrg-maker``

2. Activate directory:
    ``conda activate wrg-maker``

3. Install wtk:
    1) ``pip install git+ssh://git@github.com/moptis/wrg_maker.git``

Option 2: Clone repo (recommended for developers)
-------------------------------------------------

1. from home dir, ``git clone git@github.com/moptis/wrg_maker.git``

2. Create ``wrg-maker`` environment and install package
    1) Create a conda env: ``conda create -n wrg-maker``
    2) Run the command: ``conda activate wrg-maker``
    3) cd into the repo cloned in 1.
    4) prior to running ``pip`` below, make sure the branch is correct (install
       from main!)
    5) Install ``wrg-maker`` and its dependencies by running:
       ``pip install -e .[dev]``

3. Check that ``wrg_maker`` was installed successfully
    1) From any directory, run the following commands. This should return the
       help pages for the CLI's.

        - ``wrg_maker``
