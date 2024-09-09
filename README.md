Multi Model Server GPU
=======

This version of multi-model-server was forked from version 1.1.11. The original repo can be found at https://github.com/awslabs/multi-model-server

The scenario that brought this fork about was one that I assume other developers experienced, so I'll do my best to outline my specific use case:

The goal here was to use a Docker container formatted for multi-model hosting in Sagemaker, where the model being used relies on GPU for inference. We don't want to 
load the model from scratch for every request, which results in higher latency and exorbitant memory usage. 

When specifying 'preload_model=True', forked workers would die almost instantaneously. I determined that this was due to a memory bug happening at tradeoff 
as a side effect of traditional multiprocessing.

To accomplish this aspect with regard to the original multi-model-repo, the following was added (**during my experimentation - pre-fork**) 
at startup within the dockerd-entrypoint.py:

```python
mme_mms_config_file = pkg_resources.resource_filename(sagemaker_inference.__name__, "/etc/mme-mms.properties")
with open(mme_mms_config_file, "a") as f:
    f.write(f"\npreload_model_gpu=true\n")
```

This successfully added the argument to the config file that dictates the commandline arguments for sagemaker-inference, but the underlying usage of multi-model-server
leverages the traditional multiprocessing library when calling 

```python
worker.run_server()
```

from within mms/model_service_worker.py --> which does not handle the multithreading of applications where GPU inference is involved. 

My solution to this, was to explicitly replace the usage of

```python
multiprocessing.Process(...)
```

with an instantiation of 

```python
torch.multiprocessing.Process(...)
```

which should serve as a perfect substitute wrapper in place of the original.

I'm doing my best to keep as much the same as possible, while only making changes with regard to the forking of worker processes and the GPU.

This is currently being built and tested to handle the sharing of a loaded (**single**) GPU model across forked workers on the same EC2 instance; although, this could 
feasibly be re-tweaked to handle multi-GPU instances as well using the same libraries.


=======

| ubuntu/python-2.7 | ubuntu/python-3.6 |
|---------|---------|
| ![Python3 Build Status](https://codebuild.us-east-1.amazonaws.com/badges?uuid=eyJlbmNyeXB0ZWREYXRhIjoicGZ6dXFmMU54UGxDaGsxUDhXclJLcFpHTnFMNld6cW5POVpNclc4Vm9BUWJNamZKMGdzbk1lOU92Z0VWQVZJTThsRUttOW8rUzgxZ2F0Ull1U1VkSHo0PSIsIml2UGFyYW1ldGVyU3BlYyI6IkJJaFc1QTEwRGhwUXY1dDgiLCJtYXRlcmlhbFNldFNlcmlhbCI6MX0%3D&branch=master) | ![Python2 Build Status](https://codebuild.us-east-1.amazonaws.com/badges?uuid=eyJlbmNyeXB0ZWREYXRhIjoiYVdIajEwVW9uZ3cvWkZqaHlaRGNUU2M0clE2aUVjelJranJoYTI3S1lHT3R5THJXdklzejU2UVM5NWlUTWdwaVVJalRwYi9GTnJ1aUxiRXIvTGhuQ2g0PSIsIml2UGFyYW1ldGVyU3BlYyI6IjArcHVCaFgvR1pTN1JoSG4iLCJtYXRlcmlhbFNldFNlcmlhbCI6MX0%3D&branch=master) |

Multi Model Server (MMS) is a flexible and easy to use tool for serving deep learning models trained using any ML/DL framework.

Use the MMS Server CLI, or the pre-configured Docker images, to start a service that sets up HTTP endpoints to handle model inference requests.

A quick overview and examples for both serving and packaging are provided below. Detailed documentation and examples are provided in the [docs folder](docs/README.md).

Join our [<img src='docs/images/slack.png' width='20px' /> slack channel](https://join.slack.com/t/mms-awslabs/shared_invite/zt-6cv1kx46-MBTOPLNDwmyBynEvFBsNkQ) to get in touch with development team, ask questions, find out what's cooking and more!

## Contents of this Document
* [Quick Start](#quick-start)
* [Serve a Model](#serve-a-model)
* [Other Features](#other-features)
* [External demos powered by MMS](#external-demos-powered-by-mms)
* [Contributing](#contributing)


## Other Relevant Documents
* [Latest Version Docs](docs/README.md)
* [v0.4.0 Docs](https://github.com/awslabs/multi-model-server/blob/v0.4.0/docs/README.md)
* [Migrating from v0.4.0 to v1.0.0](docs/migration.md)

## Quick Start
### Prerequisites
Before proceeding further with this document, make sure you have the following prerequisites.
1. Ubuntu, CentOS, or macOS. Windows support is experimental. The following instructions will focus on Linux and macOS only.
1. Python     - Multi Model Server requires python to run the workers.
1. pip        - Pip is a python package management system.
1. Java 8     - Multi Model Server requires Java 8 to start. You have the following options for installing Java 8:

    For Ubuntu:
    ```bash
    sudo apt-get install openjdk-8-jre-headless
    ```

    For CentOS:
    ```bash
    sudo yum install java-1.8.0-openjdk
    ```

    For macOS:
    ```bash
    brew tap homebrew/cask-versions
    brew update
    brew cask install adoptopenjdk8
    ```

### Installing Multi Model Server with pip

#### Setup

**Step 1:** Setup a Virtual Environment

We recommend installing and running Multi Model Server in a virtual environment. It's a good practice to run and install all of the Python dependencies in virtual environments. This will provide isolation of the dependencies and ease dependency management.

One option is to use Virtualenv. This is used to create virtual Python environments. You may install and activate a virtualenv for Python 2.7 as follows:

```bash
pip install virtualenv
```

Then create a virtual environment:
```bash
# Assuming we want to run python2.7 in /usr/local/bin/python2.7
virtualenv -p /usr/local/bin/python2.7 /tmp/pyenv2
# Enter this virtual environment as follows
source /tmp/pyenv2/bin/activate
```

Refer to the [Virtualenv documentation](https://virtualenv.pypa.io/en/stable/) for further information.

**Step 2:** Install MXNet
MMS won't install the MXNet engine by default. If it isn't already installed in your virtual environment, you must install one of the MXNet pip packages.

For CPU inference, `mxnet-mkl` is recommended. Install it as follows:

```bash
# Recommended for running Multi Model Server on CPU hosts
pip install mxnet-mkl
```

For GPU inference, `mxnet-cu92mkl` is recommended. Install it as follows:

```bash
# Recommended for running Multi Model Server on GPU hosts
pip install mxnet-cu92mkl
```

**Step 3:** Install or Upgrade MMS as follows:

```bash
# Install latest released version of multi-model-server 
pip install multi-model-server
```

To upgrade from a previous version of `multi-model-server`, please refer [migration reference](docs/migration.md) document.

**Notes:**
* A minimal version of `model-archiver` will be installed with MMS as dependency. See [model-archiver](model-archiver/README.md) for more options and details.
* See the [advanced installation](docs/install.md) page for more options and troubleshooting.

### Serve a Model

Once installed, you can get MMS model server up and running very quickly. Try out `--help` to see all the CLI options available.

```bash
multi-model-server --help
```

For this quick start, we'll skip over most of the features, but be sure to take a look at the [full server docs](docs/server.md) when you're ready.

Here is an easy example for serving an object classification model:
```bash
multi-model-server --start --models squeezenet=https://s3.amazonaws.com/model-server/model_archive_1.0/squeezenet_v1.1.mar
```

With the command above executed, you have MMS running on your host, listening for inference requests. **Please note, that if you specify model(s) during MMS start - it will automatically scale backend workers to the number equal to available vCPUs (if you run on CPU instance) or to the number of available GPUs (if you run on GPU instance). In case of powerful hosts with a lot of compute resoures (vCPUs or GPUs) this start up and autoscaling process might take considerable time. If you would like to minimize MMS start up time you can try to avoid registering and scaling up model during start up time and move that to a later point by using corresponding [Management API](docs/management_api.md#register-a-model) calls (this allows finer grain control to how much resources are allocated for any particular model).**

To test it out, you can open a new terminal window next to the one running MMS. Then you can use `curl` to download one of these [cute pictures of a kitten](https://www.google.com/search?q=cute+kitten&tbm=isch&hl=en&cr=&safe=images) and curl's `-o` flag will name it `kitten.jpg` for you. Then you will `curl` a `POST` to the MMS predict endpoint with the kitten's image.

![kitten](docs/images/kitten_small.jpg)

In the example below, we provide a shortcut for these steps.

```bash
curl -O https://s3.amazonaws.com/model-server/inputs/kitten.jpg
curl -X POST http://127.0.0.1:8080/predictions/squeezenet -T kitten.jpg
```

The predict endpoint will return a prediction response in JSON. It will look something like the following result:

```json
[
  {
    "probability": 0.8582232594490051,
    "class": "n02124075 Egyptian cat"
  },
  {
    "probability": 0.09159987419843674,
    "class": "n02123045 tabby, tabby cat"
  },
  {
    "probability": 0.0374876894056797,
    "class": "n02123159 tiger cat"
  },
  {
    "probability": 0.006165083032101393,
    "class": "n02128385 leopard, Panthera pardus"
  },
  {
    "probability": 0.0031716004014015198,
    "class": "n02127052 lynx, catamount"
  }
]
```

You will see this result in the response to your `curl` call to the predict endpoint, and in the server logs in the terminal window running MMS. It's also being [logged locally with metrics](docs/metrics.md).

Other models can be downloaded from the [model zoo](docs/model_zoo.md), so try out some of those as well.

Now you've seen how easy it can be to serve a deep learning model with MMS! [Would you like to know more?](docs/server.md)

### Stopping the running model server
To stop the current running model-server instance, run the following command:
```bash
$ multi-model-server --stop
```
You would see output specifying that multi-model-server has stopped.

### Create a Model Archive

MMS enables you to package up all of your model artifacts into a single model archive. This makes it easy to share and deploy your models.
To package a model, check out [model archiver documentation](model-archiver/README.md)

## Recommended production deployments

* MMS doesn't provide authentication. You have to have your own authentication proxy in front of MMS.
* MMS doesn't provide throttling, it's vulnerable to DDoS attack. It's recommended to running MMS behind a firewall.
* MMS only allows localhost access by default, see [Network configuration](docs/configuration.md#configure-mms-listening-port) for detail.
* SSL is not enabled by default, see [Enable SSL](docs/configuration.md#enable-ssl) for detail.
* MMS use a config.properties file to configure MMS's behavior, see [Manage MMS](docs/configuration.md) page for detail of how to configure MMS.
* For better security, we recommend running MMS inside docker container. This project includes Dockerfiles to build containers recommended for production deployments. These containers demonstrate how to customize your own production MMS deployment. The basic usage can be found on the [Docker readme](docker/README.md).

## Other Features

Browse over to the [Docs readme](docs/README.md) for the full index of documentation. This includes more examples, how to customize the API service, API endpoint details, and more.

## External demos powered by MMS

Here are some example demos of deep learning applications, powered by MMS:

 |  |   |
|:------:|:-----------:|
| [Product Review Classification](https://thomasdelteil.github.io/TextClassificationCNNs_MXNet/) <img width="325" alt="demo4" src="https://user-images.githubusercontent.com/3716307/48382335-6099ae00-e695-11e8-8110-f692b9ecb831.png"> |[Visual Search](https://thomasdelteil.github.io/VisualSearch_MXNet/) <img width="325" alt="demo1" src="https://user-images.githubusercontent.com/3716307/48382332-6099ae00-e695-11e8-9fdd-17b5e7d6d0ec.png">|
| [Facial Emotion Recognition](https://thomasdelteil.github.io/FacialEmotionRecognition_MXNet/) <img width="325" alt="demo2" src="https://user-images.githubusercontent.com/3716307/48382333-6099ae00-e695-11e8-8bc6-e2c7dce3527c.png"> |[Neural Style Transfer](https://thomasdelteil.github.io/NeuralStyleTransfer_MXNet/) <img width="325" alt="demo3" src="https://user-images.githubusercontent.com/3716307/48382334-6099ae00-e695-11e8-904a-0906cc0797bc.png"> |

## Contributing

We welcome all contributions!

To file a bug or request a feature, please file a GitHub issue. Pull requests are welcome.
