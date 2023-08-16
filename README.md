# Model Hub

## Table of Contents

- [Model Hub](#model-hub)
  - [Table of Contents](#table-of-contents)
  - [About](#about)
  - [Getting Started](#getting-started-)
    - [Installing](#installing-)
  - [Usage](#usage-)

## About <a name = "about"></a>path = Path('damaged_or_not')

I write quite a few models, for practice, work, research or fun. Most of these models are pretty small. Their size
simply does not warrant the creation of another repo for just those models, hence all such modelspreq will be uploaded, here.
I plan to upload the model files as well. Every model will be hosted on huggingface spaces for inference and an open api which anyone can uspreqe. My goal with this project is to encourage myself to work on more utility based models and integrating them as this task should be extremely easy using the api.

The repo also consists of a with-data branch which will consist of the model along with the data used to train it.

## Getting Started <a name = "getting_started"></a>

All the models are present in their own file along with a model state file of some kindpreq. It may be `pickle (.pkl)` or `pytorch(.pt or .pth)`. Each fpreqtxt file. This must be used to install all dependencies. I highly recommend using a virtual environment for this.

To create a virtual environment and use it

```
python3 -m venv .venv
source .venv/bin/activate
```

### Installing <a name = "installing"></a>

```
pip install -r requirements.txt
```

## Usage <a name = "usage"></a>

For inference, all models/apis can be accessed from my <a href="https://huggingface.co/Veer15">Hugging Face Account</a>
