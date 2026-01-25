# SSH-Net: Robust Polymorphic River Extraction from Sentinel-2 Imagery

# Abstract
<img width="1937" height="575" alt="image" src="https://github.com/user-attachments/assets/9dd490f0-7dc2-423e-b5f7-d13f40b00230" />

# Overview of the network structure
<img width="882" height="502" alt="image" src="https://github.com/user-attachments/assets/1375b5a1-3d91-4d67-aa27-518898503ede" />

# Partial samples of the GPMRD
<img width="768" height="515" alt="image" src="https://github.com/user-attachments/assets/210dc045-1a6d-4aa0-b159-e7288f77fc31" />

The complete GPMRD will be uploaded soon.

# Usage 

The structure of dataset
    
    |- train
    
        |- images
      
          |- 1.tif
        
          |- 2.tif
        
          |- ...
        
        |- masks
        
          |- 1.tif
          
          |- 2.tif
          
          |- ...
          
    |- valid
    
        |- images
        
          |- 3.tif
          
          |- 4.tif
          
          |- ...
          
        |- masks
        
          |- 3.tif
          
          |- 4.tif
          
          |- ...
        
    |- test
    
        |- images
        
          |- 5.tif
          
          |- 6.tif
          
          |- ...
          
        |- masks
        
          |- 5.tif
          
          |- 6.tif
          
          |- ...


Create conda environment

    conda env create -f net.yml

Training

    python train.py --config config.yml

Test

    python evaluate.py --config config.yml


# Examples of result.

<img width="1569" height="1006" alt="image" src="https://github.com/user-attachments/assets/ec35971b-c692-4cf0-a558-4799a3eef6ac" />

<img width="1566" height="1013" alt="image" src="https://github.com/user-attachments/assets/e28cddb6-4eef-4483-9c2d-bf4ca09a1f3d" />


# Test procedure record


https://github.com/user-attachments/assets/1e60c376-e5a3-4442-9e15-e7d15aab094d

