# OpenCode Agent Instructions

## Execution
- The primary script is `macOS-Hardware-info.py`.
- **Requires `sudo`**: It must be executed with `sudo` privileges to access S.M.A.R.T. disk data (`sudo python3 macOS-Hardware-info.py`).

## Dependencies
- **System Requirements**: Requires `smartctl` to be installed on the macOS system (e.g., `brew install smartmontools`).
- **Python Dependencies**: Listed in `requirements.txt`. Note: Some libraries like `coremltools` might be sensitive to specific Python versions. 

## Documentation Discrepancies
- The `README.md` references a test file `test_macOS_Hardware_info.py` that **does not exist** in the repository. Do not attempt to run it or assume it is missing due to an error.
- There is a contradiction regarding Python versions: The script's docstring specifies `Python 3.13+`, while the `README.md` badges and troubleshooting section specify `Python 3.10` or `3.11` (and warn against 3.12+). Follow the user's explicit instructions or test the environment if version-related issues arise.
