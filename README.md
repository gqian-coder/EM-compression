# EM-compression (MGARD-X 4D STEM)

This folder contains a small C++ driver (`mgard_4dStem.cpp`) that compresses 4D STEM data using MGARD-X.

## Build (Frontier / ROCm)

A working example build script is provided in `build_script.sh`.

1. Edit the install prefixes inside `build_script.sh`:
   - `mgard_install_dir=/path/to/MGARD/install`
   - `adios_install_dir=/path/to/ADIOS2/install` (only used via `CMAKE_PREFIX_PATH` in the script)
2. Build:

```bash
./build_script.sh
```

This configures CMake into `build/` and produces the executable:

```bash
./build/mgard_4dStem
```

## Run

Example command (from `run_cmd.txt`):

```bash
./build/mgard_4dStem ../../Spectrum_Image_data_reorder_256x256x256x256.bin -1 -1 0.1 4
```

Arguments:

```text
mgard_4dStem <filename> <px> <py> <tol> <ndim>
```

- `filename`: input binary file (float32)
- `px`, `py`: slice selectors
  - `-1 -1` compresses all slices for the selected `ndim` mode
  - other values select a specific slice (see logic in `mgard_4dStem.cpp`)
- `tol`: absolute error bound
- `ndim`: compression dimensionality (supported values in this driver: `2`, `3`, `4`, or `0`)

## Notes

- The CMake config links against MGARD and HIP (`find_package(mgard REQUIRED)`, `find_package(hip REQUIRED)`).
- The code prints GPU timing (MGARD-X compress/decompress) and wall timings.
- `build/` is intentionally ignored by git via `.gitignore`.
