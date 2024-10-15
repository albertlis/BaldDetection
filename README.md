# BaldDetection

This project uses the CelebFaces dataset to detect bald individuals. We use Poetry for package management and DVC for data and model management.

## Prerequisites

- Python 3.11
- Git
- Poetry
- DVC

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-repo.git
cd your-repo
```

### 2. Install Poetry

Refer to the installation guides:
- [Pipx Installation](https://pipx.pypa.io/stable/installation/)
- [Poetry Installation](https://python-poetry.org/docs/)

### 3. Install Dependencies

To install all project dependencies, run:
Use `--no-cache` if you encounter memory errors.

```bash
poetry install
```

### 4. Activate the Environment

To activate the environment, use:

```bash
poetry shell
```

### 5. Configure DVC

Initialize DVC in the project:

```bash
dvc init
```

Download the dataset from:
[Google Drive](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg)
and Kaggle cvs annotations

Add the dataset to DVC:

```bash
dvc add path/to/dataset
```

Update the Git repository:

```bash
git add dataset.dvc .gitignore
git commit -m "Add CelebA dataset"
```

Install TurboJPEG:
[TurboJPEG Releases](https://github.com/libjpeg-turbo/libjpeg-turbo/releases)