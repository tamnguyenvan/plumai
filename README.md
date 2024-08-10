# PlumAI - Deploy and Run AI Models


![Banner](./assets/banner.png)

**PlumAI** is a CLI tool that allows you to deploy and run AI models to [Modal](https://modal.com). It's completely free and open-source.

## Table of Contents
- [Changelog](#changelog)
- [Supported Platforms](#supported-platforms)
- [Installation](#installation)
- [Usage](#usage)
  - [Run a Flux Model](#run-a-flux-model)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Contributing](#contributing)
- [Contact](#contact)

# Changelog

## [Aug 10, 2024] - 🎉 Added support for Flux models.

- Added support for Flux models: `flux.1-dev` and `flux.1-schnell`.


## Supported Platforms

PlumAI currently supports Linux, macOS, and Windows.

## Installation

To install PlumAI, follow these steps:

1. **Verify System Requirements**:
- **Python 3**: Ensure Python 3 is installed on your system.

2. **Install PlumAI**:

```bash
pip install plumai
```

This command installs PlumAI directly from the GitHub repository, including all necessary dependencies.

## Usage
After installation, you can use PlumAI with the following commands:

### Run a Flux model


```bash
plumai run [MODEL_NAME] [GPU_TYPE] [-f | --force]

# For example:
plumai run flux.1-dev A10G
```


- `MODEL_NAME`: Specify the model name to run. Options are:
  - flux.1-dev (default)
  - flux.1-schnell


- `GPU_TYPE`: Specify the GPU type to use. Options are:
  - T4 (not recommended)
  - A10G (default)
  - A100

- `f | --force`: Use this option to force redeployment of the model, ignoring the cache. If the model has already been deployed and cached, using `--force` will trigger a new deployment process.


## Troubleshooting
- Git or Python3 not installed:
If you encounter errors related to missing Git or Python3, install them using your system's package manager. For example, on Debian-based systems:

```bash
sudo apt-get update
sudo apt-get install -y git python3 python3-pip
```

- Modal Setup Issues:
If you encounter issues with Modal, ensure it is properly configured and installed. Refer to Modal's documentation for setup instructions.

## License
This project is licensed under the MIT License.

## Contributing
Contributions are welcome! Please submit issues or pull requests to the repository here.

## Contact
For questions or support, please contact me via [Twitter/X](https://x.com/tamnvvn)