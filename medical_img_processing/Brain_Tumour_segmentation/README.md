# Brain tumour segmentation
You can view the detailed tear down analysis of this here : https://maroon-profit-6d5.notion.site/Intel-One-API-Medical-image-processing-application-e5c350a6a1b143c48bed05a6d1f73fc0?pvs=4

## Visualize Brain tumour here : 
- [Show me the brain tumour visualization](https://github.com/Rahul-Talari/Intel-Api-hackathon/medical_img_processing/Brain_Tumour_segmentation
/scatter_plot.png.html)

## Code wall time Benchmarking
https://www.notion.so/Intel-One-API-Medical-image-processing-application-e5c350a6a1b143c48bed05a6d1f73fc0?pvs=4#fb9d3da542f940f695b9d53d4c73a845
![wall_time](https://github.com/Rahul-Talari/Intel-Api-hackathon/assets/91232198/cba9045e-6539-49f5-825b-49142b1b80bb)
![wall_time_graph](https://github.com/Rahul-Talari/Intel-Api-hackathon/assets/91232198/56ea0c1e-d91e-40e8-9bf1-e9325f4e722e)


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
