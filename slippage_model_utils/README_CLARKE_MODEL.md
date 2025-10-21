# Using Updated Contamination Parameterization via Clarke Model

## 🧩 Overview
This repository includes a Python wrapper and supporting R script 
for running the **[Clarke Beta-Binomial Model](https://link.springer.com/article/10.1007/s13253-023-00566-x#Sec8)**.
The Python function `run_clarke_bb_group_model()` executes the R model 
script `clarke_bb_model.R` through an R runtime managed by **Conda**. 
The setup is designed to work automatically on both **full Anaconda** and **Miniconda** installations.

---

## System Requirements
- **Windows 10/11**
- **Git** (git-scm.com)
- **Python 3.9+**
- **Conda (Anaconda or Miniconda)**

---

## Setup Instructions

### 1. Install Conda
#### Option A — Full Anaconda (recommended for developers)
If you already have Anaconda installed, open **Anaconda Prompt** and verify:
```powershell
conda --version
```

#### Option B — Miniconda (lightweight alternative)
Download from [miniconda.pydata.org](https://docs.conda.io/en/latest/miniconda.html).
After installation, open **Anaconda Prompt** and verify:
```powershell
conda --version
```

---

### 2. Create the R Environment (`rbb`)
Create an environment containing R and required R packages:
```powershell
conda create -n rbb -c conda-forge r-base r-essentials -y
```
Add additional R packages used by the model 
(the following command should cover all that's needed):
```powershell
conda install -n rbb -c conda-forge r-base r-jsonlite r-rmpfr
```
Verify Rscript runs correctly:
```powershell
conda run -n rbb Rscript --version
```
You should see the R version output (e.g., ```Rscript (R) version 4.5.1 (2025-06-13)```)

Additionally, VERIFY that there exists a directory for the `rbb` environment 
where you have Anaconda/Miniconda installed 
(will be different by user based on where you have installed Anaconda). Should be a 
directory similar to `%YOUR_PATH_TO_ANACONDA%\envs\rbb`.
Examples include:

Example 1:  `C:\Users\username1\AppData\Local\anaconda3\envs\rbb`

Example 2:  `C:\Users\username2\Anaconda3\envs\rbb`

---

### 3. Clone the Repository (skip this step if already done)

First, create a folder directory, where you want to clone the repository.

```powershell
Example
C:\Users\username1\plant-inspection-station-simulation
```

Next, retrieve the repository url.

- Navigate to the repository page in your browser.
- Look for a button or tab labeled “Code”, “Clone”, or “Clone or download.”
- Click the copy icon next to the HTTPS (or SSH) URL.
- Open a command line prompt and navigate to the folder/directory created above.
- Use that copied link to replace `%repository_url%` in the command below (remove the `%` signs)

```powershell
git clone %REPOSITORY_URL%
```

Confirm that the R script exists at:
```
plant-inspection-station-simulation\slippage_model_utils\clarke_bb_model.R
```

---

### 4. Create a Python Virtual Environment and Install Requirements

Note:  If you have one already installed, then skip this step.

```powershell
py -m venv .venv
```

You can also create the environment using conda:
 - Open the Anaconda prompt
 - Navigate to the directory to where your repo is cloned
 - Check your python version:  `python --version`
 - Create the environment: `conda create -n venv python=XXX anaconda` where "XXX" represents the version number
 - Activate the environment: `conda activate venv`


---
### 5. Activate your Virtual Environment, Upgrade Pip, Install Requirements

- Activate the environment (if not done via conda): `.\.venv\Scripts\activate`
- Upgrade pip: `python -m pip install --upgrade pip`
- Install requirements:  `pip install -r requirements.txt --timeout=10000`


---

### 6. Test R and Python Integration
Verify that Conda and Rscript work together. In **Anaconda Prompt**, run:
```powershell
conda run -n rbb Rscript -e "cat('Rscript is alive\n')"
```
You should see `Rscript is alive`.

Next, verify Python can import the wrapper.  
From either **Anaconda Prompt** or from terminal/cmd line, run:
```powershell
python -c "from slippage_model_utils.clarke_r_script_wrapper import run_clarke_bb_group_model as run; print('Wrapper imported successfully')"
```

You should see `Wrapper imported successfully`.

**YOU ARE SET!**

You should be all set to execute the Clarke Beta-Binomial Model implementation
in the `plant-inspection-station-simulation` git repository.


---

## Notes for Developers
- The wrapper uses `conda run -n rbb` to ensure consistent R environments across systems.
- JSON payloads are passed as command-line arguments; for large payloads, consider refactoring to use temp files.
- The R output must end with a valid JSON object.



---

**Maintainer:** Joseph Agor (joseph.agor@jhuapl.edu)

**Last updated:** October 2025