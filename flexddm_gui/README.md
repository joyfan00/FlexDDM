# FlexDDM

FlexDDM is a local graphical application for building, simulating, fitting, and validating decision-diffusion models. The app runs a small Python server on your own computer and opens an interactive browser-based interface for data mapping, model editing, simulation, validation, and fitting.

## Requirements

- Python 3.10 or newer
- A C++ compiler is recommended for the fastest numerical kernels, but FlexDDM can still run without the optional compiled extension.
- On macOS, install the Xcode command line tools before building the optional extension:

```bash
xcode-select --install
```

## Run From Source

Clone the repository, move into this folder, create a virtual environment, install the package, and launch the app.

### Windows

```powershell
cd path\to\FlexDDM\flexddm_gui
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e .
.\.venv\Scripts\python -m flexddm.app
```

Then open:

```text
http://127.0.0.1:8765
```

### macOS

```bash
cd /path/to/FlexDDM/flexddm_gui
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m flexddm.app
```

Then open:

```text
http://127.0.0.1:8765
```

## Build A Standalone App

FlexDDM can be packaged with PyInstaller. Build on the operating system you want to distribute for: build the Windows app on Windows, and build the Mac app on macOS.

### Windows PyInstaller Build

```powershell
cd path\to\FlexDDM\flexddm_gui
.\.venv\Scripts\python -m pip install pyinstaller pywebview pythonnet clr-loader
.\.venv\Scripts\pyinstaller FlexDDM.spec
```

The output appears in:

```text
dist\FlexDDM.exe
```

### macOS PyInstaller Build

```bash
cd /path/to/FlexDDM/flexddm_gui
source .venv/bin/activate
python -m pip install pyinstaller pywebview
pyinstaller FlexDDM-mac.spec
```

The output appears in:

```text
dist/FlexDDM.app
```

## Included Data

The source package includes one example dataset:

```text
data/ulrich2015.csv