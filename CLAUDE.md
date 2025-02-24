# Claude Helper File

## Build/Lint/Test Commands

### Python
- **Run tests**: `pytest` or `pytest test/test_specific.py -v`
- **Single test**: `pytest test/test_file.py::test_function -v`
- **Build package**: `python setup.py build`
- **Install package**: `python setup.py install` or `pip install -e .`
- **Run demo**: `python -c "import fast_cd_pyb as fcd; fcd.apps.interactive_cd_affine_handle()"`

### C++
- **CMake build**: `mkdir -p build && cd build && cmake .. && make`
- **Clean build**: `rm -rf build`

## Code Style Guidelines

### General
- Use 4-space indentation for Python
- Max line length: 100 characters
- Add type annotations to function parameters and return values
- Handle errors explicitly with proper try/except blocks

### Naming Conventions
- Classes: PascalCase (e.g., `FastCDState`)
- Functions/methods: snake_case (e.g., `compute_modes`)
- Variables: snake_case (e.g., `num_clusters`)
- Constants: UPPER_SNAKE_CASE (e.g., `MAX_ITERATIONS`)
- Private methods: _prefix (e.g., `_internal_function`)

### Imports
- Group imports: standard library, third-party, local
- Sort alphabetically within groups
- Avoid wildcard imports (`from module import *`)