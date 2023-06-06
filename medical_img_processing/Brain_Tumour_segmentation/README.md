# Brain tumour segmentation
You can view the detailed tear down analysis of this here : https://maroon-profit-6d5.notion.site/Intel-One-API-Medical-image-processing-application-e5c350a6a1b143c48bed05a6d1f73fc0?pvs=4

## Code wall time Benchmarking
https://www.notion.so/Intel-One-API-Medical-image-processing-application-e5c350a6a1b143c48bed05a6d1f73fc0?pvs=4#fb9d3da542f940f695b9d53d4c73a845
+----------------------+--------------+
| Times                |        Value |
+======================+==============+
| Data loading         |    0.234289  |
+----------------------+--------------+
| data preprocessing   |    0.117548  |
+----------------------+--------------+
| Stratification       |    0.0262754 |
+----------------------+--------------+
| Model training       |    1172.9    |
+----------------------+--------------+
| tumour predction     |    27.668    |
+----------------------+--------------+
| tumour visualization |    2.3129    |
+----------------------+--------------+
| total code wall time |    1391.44   |
+----------------------+--------------+

## Optimization pathways

1. oneAPI AI Analytics Toolkit
    - pytorch
        - ipex —> gpu —> ipex.optimize —> warmup —> inference
    - intel INC
    - sklearnex
    - modin
    - heavy.ai
    - numpy
    - dp-python
2. Intel oneAPI Base Toolkit
    - Direct download / dev cloud
    - SYCL/DPC++
    - ICX (Next Gen C++)
    - oneMKL
    - oneDNN
    - oneTBB ( Threading Building Blocks )
        - used for creating effectively threaded applications
        - effective threading across heterogenous compute devices
        - all other packages in base tool kit use this
    - oneDAL
    - oneVPL ( Video Processing Library )
        - offloading the work
    - oneCCL ( Collective Communications Library )
        - model parallelization
        - data parallelization
        - supported & optimized for intel cpus & gpus
        - pytorch doesn’t support oneCCL by default
        - one API AI analytics toolkit gets it by default

## Optimizations

- [ ]  Optimization using SYCL / DPC++
- [ ]  Graph optimization
    1. tracing & freezing the model 
    2. saving the traced model for efficient computation
- [ ]  Quantization
- [ ]  Optimization with AMX BF16 , INT8
- [ ]  Optimization with Quantization on AMX BF16
- [ ]  Channel Last ( NCHW —> NHWC conversion , CNN acceleration )
- [ ]  Auto Mixed Precision ( AMP ) with BF16
    1. 3rd Xeon - AVX 512 instruction set 
- [ ]  Operator optimization
- [ ]  OneMKL
- [ ]  MKL-DNN
- [ ]  Intel Integrated Performance Primitives (Intel IPP)
