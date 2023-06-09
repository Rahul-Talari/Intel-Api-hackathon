# intel-oneAPI

#### Team Name          : TEAM MARINES 
#### Problem Statement  : MEDICAL IMAGE PROCESSING
#### Team Leader Email  : rahultalari737@gmail.com
#### Team members email : 
#### Ramganeshlankada@gmail.com
#### mayankgla21@gmail.com
#### srinub@gvpce.ac.in ( Professionl ) 

## Prototype Video 
link : https://www.linkedin.com/posts/ram-ganesh-1b2189208_activity-7073002717509537792-rBJg/?utm_source=share&utm_medium=member_desktop

## A Brief of the Prototype:
#### General Un-optimized code 
The Medical image processing application prototype consists of 3 main categories they are : 
- Brain Tumour Segmentation 
- RSNA Pneumonia Detection
- Kidney Tumour & Stone Detection 

A well documented code for each category is present in `medical_img_processing` folder with all the 
- Data Visualizations
-  IPYNB files 
-  .py files 
-  Serialized models 
-  Images

#### Intel Optimized SYCL / DPC++ code
Once the general optimized code was done we started crafting the SYCL/DPC++ code using following path : 
Python --> C++ conversion --> DPC++ conversion --> SYCL conversion --> Intel Advisor 

All the intel optimized SYCL/DPC++ codes are present in `INTEL_ONE_API_medical_img_processing` folder. 

##### Intel Advisor Analysis

Detailed analysis of Intel Advisor for SYCL/DPC++ Pneumonia Detection is categoried into : 
- Offload Modelling
- Threading
- Vector optimization 
- Roofline Analysis

**Images** ; 
**Offload Modelling**
![main](https://github.com/Rahul-Talari/Intel-Api-hackathon/assets/91232198/1077cd3d-7c76-450c-84fc-8109234e1d4a)
![top_non_offloaded](https://github.com/Rahul-Talari/Intel-Api-hackathon/assets/91232198/feedb475-3d84-46db-a7f4-a16325dcdcbc)
![top_offloaded](https://github.com/Rahul-Talari/Intel-Api-hackathon/assets/91232198/e9af11ea-c096-455c-9078-0053f3dc8f73)

**Vectorization**
![Vectorization](https://github.com/Rahul-Talari/Intel-Api-hackathon/assets/91232198/5cbe29e6-4db9-4206-87c4-44cd0d7fcd04)

**Threading**
![Threading](https://github.com/Rahul-Talari/Intel-Api-hackathon/assets/91232198/06d66cad-c390-467f-b708-35709da1214a)
![Threading2](https://github.com/Rahul-Talari/Intel-Api-hackathon/assets/91232198/5039ea3e-03e5-4016-9ef4-3b8f06e11542)
![Threading3](https://github.com/Rahul-Talari/Intel-Api-hackathon/assets/91232198/9dcd136f-a49f-40a1-8702-c9680879761d)

**Acceleration**
![acceleration](https://github.com/Rahul-Talari/Intel-Api-hackathon/assets/91232198/67ed9a4e-91e1-41cd-ab0e-6cf68064b3a9)


## Tech Stack: 

**Machine learning Stack **
- Tensorflow
- Keras
- PyTorch
- TensorBoard 

**Intel libraries**
- Intel SYCL / DPC++ library
- Intel Base Toolkit 
- Intel Advisor 
- Intel VTune Profiler

**Application Deployment Stack**
- Streamlit 
- HTML 
- DICOM Viewer Embedding

   
## Execution Instructions:
```
git clone https://github.com/Rahul-Talari/Intel-Api-hackathon/
cd Application 
streamlit run "path/to/multiple disease pred.py"
```

Remember that the file contains statically set paths to some dependent files, scrutinize the `multiple disease pred.py` files and the change the required.
