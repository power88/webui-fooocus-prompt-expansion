# WebUI Fooocus Prompt Expansion Plugin

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE)
### **Non-Commercial Use**: The plugin is available for non-commercial use outside Fooocus.

This project extracts the prompt expansion module from Fooocus and integrates it as a plugin into the Stable Diffusion WebUI.
This plugin is just a wrapper, and its performance might be low. Pull requests to improve the code efficiency are welcome.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

- 

## Installation

To install the plugin, follow these steps:

1. Open the Stable Diffusion WebUI.
2. Navigate to `Extensions`.
3. Select `Install from URL`.
4. Enter the following URL:
   ```
   https://github.com/power88/webui-fooocus-prompt-expansion.git
   ```
5. Click `Install`.
6. Restart WebUI to download expansion model from huggingface.

## Usage

Just press enable expansion in the webui.
Apologize that the seed cannot be saved in the metadata in the output png.
We'll solve this issue as soon as possible.


## License

This project is licensed under the AGPL-3.0 License.

**Note**: If used outside Fooocus, only non-commercial use is permitted (CC-By NC 4.0).

## Contributing

We welcome contributions to improve this plugin! To contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## Acknowledgements

- Special thanks to the creators of Fooocus (lllyasviel) for the original prompt expansion module.
- Thanks to the Stable Diffusion community for their support and contributions.
- Thanks to GPT-4o for the README generate and code modification.

---

For any issues or questions, please open an issue on the [GitHub repository](https://github.com/power88/webui-fooocus-prompt-expansion/issues).