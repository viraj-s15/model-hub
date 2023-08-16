# Model Hub

## Table of Contents

- [About](#about)
- [Getting started](#getting_started)

## About <a name = "about"></a>

I write quite a few models, for practice, work, research or fun. Most of these models are pretty small. Their size 
simply does not warrant the creation of another repo for just those models, hence all such models will be uploaded, here.
I plan to upload the model files as well. Every model will be hosted on huggingface spaces for inference and an open api which anyone can use. My goal with this project is to encourage myself to work on more utility based models and integrating them as this task should be extremely easy using the api.

## Getting Started <a name = "getting_started"></a>

All the models are present in their own file along with a model state file of some kind. It may be `pickle (.pkl)` or `pytorch(.pt or .pth)`. Each file will have their own way of inference which would be described in the notebook. Each folder will have a notebook consisting of the dataloading,training and inference of the model. 

### Prerequisites

Each folder has their own requirements.txt file. This must be used to install all dependencies. I highly recommend using a virtual environment for this. 

To create a virtual environment and use it
```
python3 -m venv .venv
source .venv/bin/activate
```

### Installing

```
pip install -r requirements.txt
```

## Usage <a name = "usage"></a>
 For inference, all models/apis can be accessed from my <a href="https://huggingface.co/Veer15">Hugging Face Account</a>
