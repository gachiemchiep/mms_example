# mms example project

## Project's Structure

```bash
.
├── config.properties       : setting file of mms
├── environment.yml         : python environment
├── imgs                    : images
├── logs                    : log files of mms
│   ├── access_log.log
│   ├── mms_log.log
│   ├── mms_metrics.log
│   ├── model_log.log
│   └── model_metrics.log
├── mms                     : source code for each model
│   ├── densenet-pytorch    : densenet, pytorch
│   └── squeezenet-mxnet    : squeenet, mxnet
├── model-archives          : model archieve file for mms
│   ├── densenet121-pytorch.mar
│   └── squeezenet_v1.1.mar
└── README.md               : this file
```

## How to use

