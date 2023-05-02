# TFLITE ACAP

## Overview
This examples shows how to run TLITE models in an Axis camera using the ACAP platform.
The TFLITE models must have one output with lable scores provided as an int8 array.
Models exported from [Googels Teachable Machine](https://teachablemachine.withgoogle.com/) are aligned with this output.

The final output are ACAPs that can run camera based on:
* ARTPEC-8
* ARMv7hf TPU (P3255, AXIS Q1615 Mk III, Q1715, M4308 )
* ARMv7hf (All models without TPU.  CPU will be used)

The package can be compiled as-is.  The model included is mobilenet V2 224 but the idea is to replace this with your own model.

## Building
1. Clone this repository
2. Use [Googels Teachable Machine](https://teachablemachine.withgoogle.com/) to train your model.
3. Export the model in both TFLITE Edge TPU and TFLITE Quntization.
4. Unzip both files and place the files under app/model. 
It is recommeded to have file names that easily seperates the EdgeTPU from the Quantized model file.  The labels.txt must be called labels.txt and must have an empty last line.
5. Edit the Dockerfile line 70 and 72 with the filename you chose under app/model/ e.g. ```/opt/app/model/model_quant.tflite```.  Make sure that the EdgeTPU and Quant file are set on the correct lines based on platform.  Note that Dockerfile will copy the correct file to model/model.tflite to be included in the ACAP depending on the platform selected.
6. Compile the ACAP from tflite_1/ directory. Type:  
   ```. artpec8.sh```  
   ```. edgetpu.sh```  
   ```. armv7hf.sh```
7. Install the eap-files in appropriate camera model
 
 ## Usage
Clients may request inference using the the URL ```http://camera-ip/local/tflite/inference```.  The ACAP web page uses the same CGI to update the result every 500ms. 

Response 

```
{
  "device":"B8A44FXXXXXX",
  "timestamp":1681550011529,
  "duration":39,  //Amount of milliseconds used for inference
  "list":[
    { "label": string, "score": number 0-100},
    ...
  ]
}
```
You can set confidence to elimiate unwanted detections in the list.

## Customization
You can customize the package in name, HTML, CGI, behavior and output.

### Name
The ACAP has a package name (inference) and a Nice Name (Teachable Macine).  These names are defined in the following files
* app/manifest.json.*
* app/Makefile (row 1)
* main.c (row 21)

### User interface page
* app/html/index.html

