# RBS PIS Slippage Model Install Instructions

_This software is controlled under the Export Administration Regulations (EAR) (15 CFR 730-744). It has an ECCN EAR99 and may require an export authorization to transfer to foreign persons._


_All new files added without license headers pending OSS approval; intended to be GPL-compatible._

Johns Hopkins University Applied Physics Laboratory (APL) has extended
the simulation framework, PoPS Border, co-developed by North Carolina
State University and the USDA Animal Plant Health and Inspection
Services (APHIS).  APL has implemented additional features
and functionality that applies the USDA Risk Based Sampling (RBS)
inspection process being conducted at 
USDA plant inspection stations (PIS).

Features and Functionality Added:
- Data driven options for generating and contaminating consignments
- Additional metrics
- Graphic user interface for building scenarios, running the model, and visualizing results
- RBS strategies captured via uploaded compliance tables

_Development Notice: This software is under active development with limited functionality.  Features may be incomplete and subject to change. Results have not been validated._



---
## Things to know before You Start

### Required Software

**_Note_**:  _You can skip any aspects you already have
completed or have on your machine (e.g., if you have Python installed,
you can skip step 2 below)_

This software has been tested on **Windows 10/11**.

You will need the following **before** proceeding:

1. **Install Conda or Miniconda with Python 3.11 (Miniconda recommended)**
   - Download from: https://repo.anaconda.com/miniconda/
   - Choose this version for 64-bit Windows: "Miniconda3-py311_25.11.1-1-Windows-x86_64.exe"
   - During installation, accept default and recommended installation options

2. **Local Access To This Git Repository**
   - You can download the repository as a archive ('zip' file)
   - You can install GitHub to access this repository
   - You can install Git GUI to access this repository
   - You can use GitLab to access/download this repository as an archive

After installing the above, **restart your computer** before continuing.

---
### Using Command Windows

You will be using the **Anaconda Prompt window** to utilize conda as you create virtual environments and access the model and UI:

- **How to Access**  
  Install Conda or Miniconda as mentioned above
  Find and run "Anaconda Prompt" from your Start menu

You will be using **Command prompt** only if you decide to update the code repository through Git rather than as an archive.

- **How to Access**  
  Type "Command Prompt" into the start menu.

---
### Accessing the Code Repository

1. Using GitLab/downloaded archive
   - Open the GitLab repository link in your web browser
   - Click the blue **Code** button
   - Under "download source code" select "zip"
   - open the downloaded archive file and copy the contents in your "User" folder 
     - e.g., `"C:\Users\<your-username>"`
   - Ensure that the top level directory folder name after `<your-username>` is `plant-inspection-station-simulation`. 
   If the folder name includes any additional text at the end (e.g., `plant-inspection-station-simulation-main`), 
   you should remove the excess text (e.g., `-main`) by renaming the folder.

2. Using GitHub
   - Open the GitLab repository link in your web browser
   - Click the blue **Code** button
   - select the option "Clone with HTTPS"
   - Open **PowerShell**, then run:
```powershell
    cd C:\Users\<your-username>
    git clone <PASTE_THE_URL_HERE>
```

If you see files such as Pipfile, frontend.py, and slippage_model_utils,
you are in the correct directory.

--- 

## Initial Setup and Installation of PoPs Border and New Features
High-level, this is a three step process to install and run the model with the updated features: You will need to create a python virtual environment to run the frontend, an rscript virtual environment to run the PoPs Border slippage model, and then run the PoPs Border model through the frontend.


### Setting up the Python Virtual Environment
   1. Open **Anaconda Prompt** command window.
     - This should open into your "Username" folder. If not, navigate there through the command prompt.
```
     cd C:\Users\<your-username>
```
   2. Enter the archive folder containing the files copied/cloned above.
```
    cd plant-inspection-station-simulation
```
  3. Use conda to create a python virtual environment called 'venv'.
```
    conda create -n venv python=3.11.14 anaconda
```
**_Note_**: _You will be prompted to accept the Terms of Service (TOS) before being allowed to continue. Review these and type "y" and click enter to continue. You will also be prompted to proceed after the package list is presented; click "y" and enter again to continue._


  4. activate the Python virtual environment.
```
    conda activate venv
```
5. Install/update/verify the most up-to-date version of  'pip' within the Python virtual environment.

```
    python -m pip install --upgrade pip
```

6. Install/update/verify the required libraries from within the "Requirement.txt" document within the plant-inspection-station-simulation folder.
```
    pip install -r requirements.txt --timeout=10000
```
---
### Setting up the R-script Virtual Environment
In the same **Anaconda Prompt** window, create an R-script virtual environment called "rbb" and updated it with required libraries.

```
    conda create -n rbb -c conda-forge r-base r-essentials -y
```
```
    conda install -n rbb -c conda-forge r-base r-jsonlite r-rmpfr
```
**_Note_**: _You may be prompted to accept the Terms of Service (TOS) again before being allowed to continue. Review these and type "y" and click enter to continue. You may also be prompted to proceed again after the package list is presented; click "y" and enter again to continue._

---
### Verify Correct Virtual Environment Setup
In the same **Anaconda Prompt** window, run these scripts to verify that the virtual environments are working correctly.
1. (optional) Activate the venv environment; you can skip this step if already active. You can tell if the virtual environment is already active if the command line prompt states with "(venv)" instead of "(base)".
```
    conda activate venv
```
2. Verify that the R script runs within the venv virtual environment using conda as a bridge.
``` 
    conda run -n rbb Rscript --version
```
```
    conda run -n rbb Rscript -e "cat('Rscript is alive\n')"
```
Running this script should respond "Rscript is alive".

3. Verify that the wrapper runs successfully
```
    Conda run -n venv python -c "from slippage_model_utils.clarke_r_script_wrapper import run_clarke_bb_group_model as run; print('Wrapper imported successfully')"
```
Running this script should respond "Wrapper imported successfully". 

---
### Verifying Streamlit 
In the same **Anaconda Prompt** window, run these scripts to verify that the virtual environments are working correctly.

1. (optional) Activate the Python virtual environment (venv); you can skip this step if already active. You can tell if the virtual environment is already active if the command line prompt starts with "(venv)" instead of "(base)".
```
    conda activate venv
```
2. Verify that streamlit is installed and accessible within the virtual environment. 
```
    streamlit --version
```
---
## Activating the PoPs Border GUI
1. (optional) Activate the Python virtual environment (venv); you can skip this step if already active. You can tell if the virtual environment is already active if the command line prompt starts with "(venv)" instead of "(base)".
```
    conda activate venv
```
2. Activate the GUI via streamlit.
```
    streamlit run frontend.py
```

After running the above command, a web browser should automatically open
with the initial landing page.  If not, then it should provide a link
that you can open.

### Additional Notes for GUI Use

- All outputs (files, figures, etc.) generated by GUI
will be put into a sub-directory of the top-level directory 
called "Outputs".
- Items used/created by the "scenario builder" components
of the GUI will be placed in a sub-directory called "tmp".

**Maintainers and POCs:** Joseph Agor (joseph.agor@jhuapl.edu) and Gary Lin (Gary.Lin@jhuapl.edu)

**Last updated:** January 2026

---
---
# PoPS Border Documentation

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ncsu-landscape-dynamics/popsborder/main?urlpath=lab/tree/examples/notebooks/basic_with_command_line.ipynb)
[![CI](https://github.com/ncsu-landscape-dynamics/popsborder/workflows/CI/badge.svg)](https://github.com/ncsu-landscape-dynamics/popsborder/actions/workflows/ci.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

PoPS Border is a simulation (simulator) of contaminated consignments and
contaminant presence testing which generates synthetic shipment data and
performs inspection on them.

## Model of reality

The simulation is using the following model to understand the system:

```math
f(x) -> y
```

where _x_ represents all shipments with all information about them
such as the level of infestation, _f_ is a sampling function,
i.e. import procedure used at the port,
and _y_ represents the resulting record in the database.

Since the simulation is generating _x_, we can compute:

```math
r = g(y) / g(x)
```

where _x_ and _y_ are defined in the same way as above,
_g_ is a function giving level of infestation in each set
(e.g. number of shipments with a pest),
and _r_ is the success rate in detecting infestation
using the function _f_ from above.

## Use cases

This simulation tool can help to answer various questions about influence
of inspection protocols or pest or contaminant presence on inspection outcome.
For example, the tool can generate synthetic data representing consignments
with variations in contamination rates and test how different inspection
methods influence inspection outcomes.
See more use cases in a dedicated [documentation section](docs/use_cases.md).

The prototype of the simulation was called _pathways-simulation_ because
for some contaminants, such as pests, the main question is what
are the pathways by which the contaminants are getting across the border.

## Examples

An example of how the simulation interface works is in
[this Jupyter notebook](examples/notebooks/basic_with_command_line.ipynb).

To run the code without installing anything use Binder:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ncsu-landscape-dynamics/popsborder/main?urlpath=lab/tree/examples/notebooks/basic_with_command_line.ipynb)

If you are not familiar with Binder, see
[our short intro](docs/binder.md).

Documentation is included in the [docs](docs/) directory.
[command line interface](docs/cli.md)
and [Consignment configuration](docs/consignments.md)
pages are good ones to start with.

## Authors

- Vaclav Petras, NCSU Center for Geospatial Analytics
- Kellyn P. Montgomery, NCSU Center for Geospatial Analytics
- Anna Petrasova, NCSU Center for Geospatial Analytics

## License

The simulation code is open source under GNU GPL >=v2
(see the LICENSE file for details).

## Acknowledgment and Disclaimer

This research is funded by USDA APHIS. The findings do not necessarily
represent the views of USDA APHIS.

Please note that this is a simulation and it needs to be calibrated
to give any realistic or actionable results. Results presented here
are examples for demonstration purposes only.
