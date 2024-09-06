**Currently in the works, first commit coming soon --> check back later**

This version of multi-model-server was (**is being**) forked from version 1.1.11. The original repo can be found at https://github.com/awslabs/multi-model-server

The scenario that brought this fork about was one that I assume other developers experienced, so I'll do my best to outline my specific use case as follows:

The goal here is to use a Docker container formatted for multi-model hosting, where the model being used relies on GPU for inference. We don't want to 
load the model from scratch for every request, which results in higher latency and exorbitant memory usage. To accomplish this aspect with regard to the original 
multi-model-repo, the following was added (**during my experimentation - pre-fork**) at startup within the dockerd-entrypoint.py:

```python
mme_mms_config_file = pkg_resources.resource_filename(sagemaker_inference.__name__, "/etc/mme-mms.properties")
with open(mme_mms_config_file, "a") as f:
    f.write(f"\npreload_model_gpu=true\n")
```

This successfully added the argument to the config file that dictates the commandline arguments to run the server at build time, but the underlying usage of multi-model-server
leverages the traditional multiprocessing library when calling 

```python

worker.run_server()

```

from within mms/model_service_worker.py --> which does not handling the multithreading of applications where GPU inference is involved. 

My solution to this, was to explicitly replace the usage of 

```python
multiprocessing.Process(...)
```

with an instantiation of torch.multiprocessing.

I'm doing my best to keep as much the same as possible, while only making changes with regard to the forking of worker processes and the GPU. 

Check back later for an update, this project was just started.

---
Original credit for the multi-model-server library: Amazon.com, Inc
