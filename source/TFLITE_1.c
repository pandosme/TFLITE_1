/*
 *	Fred Juhlin 2023
 *	Optimized for quantized TFLITE files with one output where lable scores are provided as an int8 array
 *	
 *	Based on https://github.com/AxisCommunications/acap3-examples/tree/main/object-detection
*/

#include <stdlib.h>
#include <stdio.h>
#include <glib.h>
#include <glib/gi18n.h>
#include <string.h>
#include <syslog.h>
#include <axsdk/axparameter.h>
#include <sys/time.h>
#include <sys/types.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "imgconverter.h"
#include "imgprovider.h"
#include "imgutils.h"
#include "larod.h"
#include "vdo-frame.h"
#include "vdo-types.h"

#include "cJSON.h"
#include "DEVICE.h"
#include "HTTP.h"
#include "FILE.h"
#include "STATUS.h"
#include "PARSER.h"

#define LOG(fmt, args...)    { syslog(LOG_INFO, fmt, ## args); printf(fmt, ## args);}
#define LOG_WARN(fmt, args...)    { syslog(LOG_WARNING, fmt, ## args); printf(fmt, ## args);}
//#define LOG_TRACE(fmt, args...)    { syslog(LOG_INFO, fmt, ## args); printf(fmt, ## args); }
#define LOG_TRACE(fmt, args...)    {}

// Hardcode to use three image "color" channels (eg. RGB).
const unsigned int CHANNELS = 3;


unsigned modelWidth = 224;
unsigned modelHeigth = 224;
double confidenceLevel = 60;
unsigned int streamWidth = 0;
unsigned int streamHeight = 0;

char modelFilePath[128];
char labelsFilePath[128];
size_t numberOfLabels = 0; // Will be parsed from the labels file

// Name patterns for the temp file we will create.
char CONV_INP_FILE_PATTERN[] = "/tmp/larod.in.test-XXXXXX";
char CONV_OUT1_FILE_PATTERN[] = "/tmp/larod.out1.test-XXXXXX";

larodModel* model = NULL;
ImgProvider_t* provider = NULL;
larodError* error = NULL;
larodConnection* conn = NULL;
larodTensor** inputTensors = NULL;
size_t numInputs = 0;
larodTensor** outputTensors = NULL;
size_t numOutputs = 0;
larodInferenceRequest* infReq = NULL;
void* larodInputAddr = MAP_FAILED;
void* larodOutput1Addr = MAP_FAILED;
int larodModelFd = -1;
int larodInputFd = -1;
int larodOutput1Fd = -1;

cJSON* labels = 0;
cJSON* TFLITE_Settings = 0;
const char* ACAP_PACKAGE = 0;

cJSON*
parseLabels(const char *labelsPath ) {
    const size_t LINE_MAX_LEN = 120;
    bool ret = false;
    char* labelsData = NULL;  // Buffer containing the label file contents.

	LOG_TRACE("%s:\n",__func__);

    struct stat fileStats = {0};
    if (stat(labelsPath, &fileStats) < 0) {
        LOG_WARN( "%s: Unable to get stats for label file %s: %s\n", __func__, labelsPath, strerror(errno));
        return 0;
    }

    if (fileStats.st_size > (10 * 1024 * 1024)) {
        LOG_WARN( "%s: failed sanity check on labels file size\n", __func__);
        return 0;
    }

    int labelsFd = open(labelsPath, O_RDONLY);
    if (labelsFd < 0) {
        LOG_WARN( "%s: Could not open labels file %s: %s\n", __func__, labelsPath, strerror(errno));
        return false;
    }

    size_t labelsFileSize = (size_t) fileStats.st_size;
    // Allocate room for a terminating NULL char after the last line.
    labelsData = malloc(labelsFileSize + 50);
    if (labelsData == NULL) {
        LOG_WARN( "%s: Failed allocating lbuffer: %s\n", __func__, strerror(errno));
		free(labelsData);
		return cJSON_CreateArray();
    }

    ssize_t numBytesRead = -1;
    size_t totalBytesRead = 0;
    char* fileReadPtr = labelsData;
    while (totalBytesRead < labelsFileSize) {
        numBytesRead = read(labelsFd, fileReadPtr, labelsFileSize - totalBytesRead);
        if (numBytesRead < 1) {
            LOG_WARN( "%s: Failed reading from labels file: %s\n", __func__, strerror(errno));
			free(labelsData);
			return 0;
        }
        totalBytesRead += (size_t) numBytesRead;
        fileReadPtr += numBytesRead;
    }
	cJSON* list = PARSER_SplitToJSON( labelsData,'\n');
	free( labelsData );
	return list;
}

/**
 * @brief Creates a temporary fd and truncated to correct size and mapped.
 *
 * This convenience function creates temp files to be used for input and output.
 *
 * @param fileName Pattern for how the temp file will be named in file system.
 * @param fileSize How much space needed to be allocated (truncated) in fd.
 * @param mappedAddr Pointer to the address of the fd mapped for this process.
 * @param Pointer to the generated fd.
 * @return Positive errno style return code (zero means success).
 */
bool createAndMapTmpFile(char* fileName, size_t fileSize, void** mappedAddr, int* convFd) {

    int fd = mkstemp(fileName);
    if (fd < 0) {
        LOG_WARN( "%s: Unable to open temp file %s: %s\n", __func__, fileName, strerror(errno));
        goto error;
    }
    // Allocate enough space in for the fd.
    if (ftruncate(fd, (off_t) fileSize) < 0) {
        LOG_WARN( "%s: Unable to truncate temp file %s: %s\n", __func__, fileName, strerror(errno));
        goto error;
    }

    // Remove since we don't actually care about writing to the file system.
    if (unlink(fileName)) {
        LOG_WARN( "%s: Unable to unlink from temp file %s: %s\n", __func__, fileName, strerror(errno));
        goto error;
    }

    // Get an address to fd's memory for this process's memory space.
    void* data = mmap(NULL, fileSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

    if (data == MAP_FAILED) {
        LOG_WARN( "%s: Unable to mmap temp file %s: %s\n", __func__, fileName, strerror(errno));
        goto error;
    }

    *mappedAddr = data;
    *convFd = fd;

    return true;

error:
    if (fd >= 0) {
        close(fd);
    }


    return false;
}

/**
 * @brief Sets up and configures a connection to larod, and loads a model.
 *
 * Opens a connection to larod, which is tied to larodConn. After opening a
 * larod connection the chip specified by larodChip is set for the
 * connection. Then the model file specified by larodModelFd is loaded to the
 * chip, and a corresponding larodModel object is tied to model.
 *
 * @param larodChip Speficier for which larod chip to use.
 * @param larodModelFd Fd for a model file to load.
 * @param larodConn Pointer to a larod connection to be opened.
 * @param model Pointer to a larodModel to be obtained.
 * @return false if error has occurred, otherwise true.
 */
bool setupLarod(const int larodModelFd, larodConnection** larodConn, larodModel** model) {
    larodError* error = NULL;
    larodConnection* conn = NULL;
    larodModel* loadedModel = NULL;
    bool ret = false;

	LOG_TRACE("%s:\n",__func__);

    // Set up larod connection.
    if (!larodConnect(&conn, &error)) {
        LOG_WARN( "%s: Could not connect to larod: %s\n", __func__, error->msg);
        goto end;
    }

    // Test various chip configuration
	//LAROD_CHIP_TFLITE_CPU, LAROD_CHIP_TPU, LAROD_CHIP_TFLITE_CPU, LAROD_CHIP_TFLITE_ARTPEC8DLPU	
    if (larodSetChip(conn, 4, &error)) {
		STATUS_SetString( "model", "architecture", "EdgeTPU" );	
	} else {
		larodClearError(&error);
		if (larodSetChip(conn, 12, &error)) {
			STATUS_SetString( "model", "architecture", "ARTPEC-8" );	
		} else {
			larodClearError(&error);
			if (larodSetChip(conn, 2, &error)) {
				STATUS_SetString( "model", "architecture", "CPU" );	
			} else {
				STATUS_SetString("model","status", "No compatible architecture found");
				LOG_WARN("No Larod compatible chip found\n");
				goto error;
			}
		}
    }

    loadedModel = larodLoadModel(conn, larodModelFd, LAROD_ACCESS_PRIVATE, ACAP_PACKAGE, &error);
    if (!loadedModel) {
		STATUS_SetString("model","status", "Unable to load model");
        LOG_WARN( "%s: Unable to load model: %s\n", __func__, error->msg);
        goto error;
    }

    *larodConn = conn;
    *model = loadedModel;

    ret = true;

    goto end;

error:
    if (conn) {
        larodDisconnect(&conn, NULL);
    }

end:
    if (error) {
        larodClearError(&error);
    }

    return ret;
}

int inferenceRunning = 0;

cJSON*
TFLITE_Inference() {

	struct timeval startTs, endTs;
	unsigned int elapsedMs = 0;

	if( !TFLITE_Settings ) {
		LOG_WARN("%s: TFLITE_Settings is NULL\n", __func__ );
		return 0;
	}

	//Check that everything is initialized
	if( !STATUS_Bool( "model", "state" ) ) {
		return 0;
	}


	if( inferenceRunning )
		return 0;
	inferenceRunning = 1;

	if( !provider) {
		STATUS_SetBool("model","state",0);
		STATUS_SetString("model","status","No image provider");
		return 0;
	}

	// Get latest frame from image pipeline.
	VdoBuffer* buf = getLastFrameBlocking(provider);
	if (!buf) {
		LOG_WARN( "%s: No image avaialable\n", __func__ );
		STATUS_SetBool("model","state",0);
		STATUS_SetString("model","status","No image provider");
		inferenceRunning = 0;
		return 0;
	}

	// Get data from latest frame.
	uint8_t* nv12Data = (uint8_t*) vdo_buffer_get_data(buf);

	// Covert image data from NV12 format to interleaved uint8_t RGB format.
	gettimeofday(&startTs, NULL);


	if (!convertCropScaleU8yuvToRGB(nv12Data, streamWidth, streamHeight, (uint8_t*) larodInputAddr, modelWidth,	modelHeigth)) {
		LOG_WARN( "%s: Failed img scale/convert in convertCropScaleU8yuvToRGB() (continue anyway)\n", __func__);
	}

	gettimeofday(&endTs, NULL);

	elapsedMs = (unsigned int) (((endTs.tv_sec - startTs.tv_sec) * 1000) +
								((endTs.tv_usec - startTs.tv_usec) / 1000));

	if (lseek(larodOutput1Fd, 0, SEEK_SET) == -1) {
		LOG_WARN( "%s: Unable to rewind output file position: %s\n", __func__, strerror(errno));
		return 0;
	}

	inferenceRunning = 1;

	gettimeofday(&startTs, NULL);
	
	if (!larodRunInference(conn, infReq, &error)) {
		LOG_WARN( "%s: Unable to run inference on model %s: %s (%d)\n", __func__, modelFilePath, error->msg, error->code);
		larodClearError(&error);		
		inferenceRunning = 0;
		return 0;
	}

	gettimeofday(&endTs, NULL);

	elapsedMs = (unsigned int) (((endTs.tv_sec - startTs.tv_sec) * 1000) + ((endTs.tv_usec - startTs.tv_usec) / 1000));

	cJSON* payload = cJSON_CreateObject();
	cJSON_AddStringToObject( payload,"device", DEVICE_Prop("serial"));
	cJSON_AddNumberToObject( payload,"timestamp", DEVICE_Timestamp());
	cJSON_AddNumberToObject( payload,"duration", elapsedMs);
	cJSON* list = cJSON_CreateArray();
	cJSON_AddItemToObject( payload,"list", list);

	uint8_t* outputPtr = (uint8_t*) larodOutput1Addr;
	
	int i;
	for( i = 0; i < numberOfLabels; i++ ) {
		double score = (double)(*((uint8_t*) (outputPtr + i)) / 255.0 * 100);  //Turn 0-255 to 0-100%
		if( score >=  confidenceLevel && labels && labels->type != cJSON_NULL ) {
			cJSON* item = cJSON_CreateObject();
			cJSON_AddStringToObject( item,"label",cJSON_GetArrayItem(labels,i)->valuestring );
			cJSON_AddNumberToObject( item,"score", (int)score );
			cJSON_AddItemToArray(list,item);
		}
	}
	
	returnFrame(provider, buf);
	inferenceRunning = 0;	
	LOG_TRACE("%s: Exit\n",__func__);
	return payload;
}


static void
TFLITE_HTTP_Settings(const HTTP_Response response,const HTTP_Request request) {
	
	if( !TFLITE_Settings ) {
		LOG_WARN("%s: TFLITE_Settings is NULL\n",__func__ );
		HTTP_Respond_Error( response, 400, "Settings corrupt");
		return;
	}

	const char *json = HTTP_Request_Param( request, "json");
	if( !json )
		json = HTTP_Request_Param( request, "set");
	if( !json ) {
		HTTP_Respond_JSON( response, TFLITE_Settings );
		return;
	}

	LOG_TRACE("%s: %s\n",__func__,json);

	cJSON *params = cJSON_Parse(json);
	if(!params) {
		HTTP_Respond_Error( response, 400, "Invalid JSON data");
		return;
	}

	cJSON* param = params->child;
	while(param) {
		if( cJSON_GetObjectItem(TFLITE_Settings,param->string ) )
			cJSON_ReplaceItemInObject(TFLITE_Settings,param->string,cJSON_Duplicate(param,1) );
		param = param->next;
	}
	cJSON_Delete(params);

	confidenceLevel = cJSON_GetObjectItem(TFLITE_Settings,"confidence")?cJSON_GetObjectItem(TFLITE_Settings,"confidence")->valuedouble:60.0;
	
	FILE_Write( "localdata/model.json", TFLITE_Settings);
	LOG_TRACE("HTTP Exit\n");
	HTTP_Respond_Text( response, "OK" );
}

void 
TFLITE_Close() {
	
    if (provider) {
		stopFrameFetch(provider);
        destroyImgProvider(provider);
	}
    
	if( model )
		larodDestroyModel(&model);

    if (conn)
        larodDisconnect(&conn, NULL);
    
    if (larodModelFd >= 0)
        close(larodModelFd);

    if (larodInputAddr != MAP_FAILED)
        munmap(larodInputAddr, modelWidth * modelHeigth * CHANNELS);

    if (larodInputFd >= 0)
        close(larodInputFd);

    if (larodOutput1Addr != MAP_FAILED)
        munmap(larodOutput1Addr, numberOfLabels );

    if (larodOutput1Fd >= 0)
        close(larodOutput1Fd);

    larodDestroyInferenceRequest(&infReq);
    larodDestroyTensors(&inputTensors, numInputs);
    larodDestroyTensors(&outputTensors, numOutputs);
    larodClearError(&error);
    
	STATUS_SetString( "model", "status", "Not avaialble" );
	STATUS_SetBool( "model", "state", 0 );	
	STATUS_SetString( "model", "acrhitecture", "Undefined" );	

}

cJSON*
TFLITE( const char* package ) {
	LOG_TRACE("%s: \n",__func__);
	ACAP_PACKAGE = package;
	STATUS_SetString( "model", "status", "Initializing" );
	STATUS_SetBool( "model", "state", 0 );	
	STATUS_SetString( "model", "architecture", "Undefined" );	

	sprintf(modelFilePath,"/usr/local/packages/%s/model/model.tflite", package);
	sprintf(labelsFilePath,"/usr/local/packages/%s/model/labels.txt", package);

	TFLITE_Settings = FILE_Read( "html/config/model.json" );
	if(!TFLITE_Settings)
		TFLITE_Settings = cJSON_CreateObject();

	cJSON* savedSettings = FILE_Read( "localdata/settings.json" );
	if( savedSettings ) {
		cJSON* prop = savedSettings->child;
		while(prop) {
			if( cJSON_GetObjectItem(TFLITE_Settings,prop->string ) )
				cJSON_ReplaceItemInObject(TFLITE_Settings,prop->string,cJSON_Duplicate(prop,1) );
			prop = prop->next;
		}
		cJSON_Delete(savedSettings);
	}

	confidenceLevel = cJSON_GetObjectItem(TFLITE_Settings,"confidence")?cJSON_GetObjectItem(TFLITE_Settings,"confidence")->valuedouble:60.0;
	modelWidth = cJSON_GetObjectItem(TFLITE_Settings,"modelWidth")?cJSON_GetObjectItem(TFLITE_Settings,"modelWidth")->valueint:224;
	modelHeigth = cJSON_GetObjectItem(TFLITE_Settings,"modelHeigth")?cJSON_GetObjectItem(TFLITE_Settings,"modelHeigth")->valueint:224;
	
	if( !cJSON_GetObjectItem(TFLITE_Settings,"labels") )
		cJSON_AddItemToObject(TFLITE_Settings,"labels",parseLabels(labelsFilePath));
	if( cJSON_GetObjectItem(TFLITE_Settings,"labels")->type == cJSON_NULL )
		cJSON_ReplaceItemInObject(TFLITE_Settings,"labels",parseLabels(labelsFilePath));
	labels = cJSON_GetObjectItem(TFLITE_Settings,"labels");
	numberOfLabels = labels?cJSON_GetArraySize(labels):0;

	if( numberOfLabels == 0 ){
		STATUS_SetBool("model","state",0);
		STATUS_SetString("model","status","No labels for this model");
        TFLITE_Close();
		return 0;
	}

    if (!chooseStreamResolution(modelWidth, modelHeigth, &streamWidth,&streamHeight)) {
        LOG_WARN( "%s: Failed choosing stream resolution\n", __func__);
		STATUS_SetBool("model","state",0);
		STATUS_SetString("model","status","No valid stream resolutions");
        TFLITE_Close();
		return 0;
    }

    provider = createImgProvider(streamWidth, streamHeight, 2, VDO_FORMAT_YUV);
    if (!provider) {
		LOG_WARN( "%s: Failed to create ImgProvider\n", __func__);
		STATUS_SetBool("model","state",0);
		STATUS_SetString("model","status","Failed to create image provider");
        TFLITE_Close();
		return 0;
    }

    larodModelFd = open(modelFilePath, O_RDONLY);
    if (larodModelFd < 0) {
        LOG_WARN( "%s: Unable to open model file %s: %s\n", __func__,modelFilePath,  strerror(errno));
        TFLITE_Close();
		STATUS_SetBool("model","state",0);
		STATUS_SetString("model","status","Model file does not exist");
		return 0;
    }
    if (!setupLarod(larodModelFd, &conn, &model)) {
        TFLITE_Close();
		STATUS_SetBool("model","state",0);
		STATUS_SetString("model","status","Failed setting up architecture");
		return 0;
    }

    // Allocate space for input tensor
    if (!createAndMapTmpFile(CONV_INP_FILE_PATTERN, modelWidth * modelHeigth * CHANNELS, &larodInputAddr, &larodInputFd)) {
		STATUS_SetString( "model", "status", "Allocation failed" );
        TFLITE_Close();
		STATUS_SetBool("model","state",0);
		STATUS_SetString("model","status","Input data allocation failed");
		return 0;
    }
    // Allocate space for output tensor 1 (Locations)
    if (!createAndMapTmpFile(CONV_OUT1_FILE_PATTERN, numberOfLabels,  &larodOutput1Addr, &larodOutput1Fd)) {
		STATUS_SetString( "model", "status", "Allocation failed" );
        TFLITE_Close();
		STATUS_SetBool("model","state",0);
		STATUS_SetString("model","status","Output data allocation failed");
		return 0;
    }
    inputTensors = larodCreateModelInputs(model, &numInputs, &error);
    if (!inputTensors) {
		STATUS_SetString( "model", "status", "Failed retrieving input tensors" );
        LOG_WARN( "Failed retrieving input tensors: %s\n", error->msg);
        TFLITE_Close();
		STATUS_SetBool("model","state",0);
		STATUS_SetString("model","status","Failed initializing input tensor");
		return 0;
    }
    if (!larodSetTensorFd(inputTensors[0], larodInputFd, &error)) {
		STATUS_SetString( "model", "status", "Failed setting input tensor" );
        LOG_WARN( "%s: Failed setting input tensor fd: %s\n", __func__,error->msg);
        TFLITE_Close();
		STATUS_SetBool("model","state",0);
		STATUS_SetString("model","status","Failed initializing input tensor");
		return 0;
    }
    outputTensors = larodCreateModelOutputs(model, &numOutputs, &error);
    if (!outputTensors) {
		STATUS_SetString( "model", "status", "Failed retrieving output tensors" );
        LOG_WARN( "%s: Failed retrieving output tensors: %s\n", __func__, error->msg);
        TFLITE_Close();
		STATUS_SetBool("model","state",0);
		STATUS_SetString("model","status","Failed initializing output tensor");
		return 0;
    }

    if (!larodSetTensorFd(outputTensors[0], larodOutput1Fd, &error)) {
		STATUS_SetString( "model", "status", "Failed setting output tensor" );
        LOG_WARN( "%s: Failed setting output tensor fd: %s\n", __func__, error->msg);
        TFLITE_Close();
		STATUS_SetBool("model","state",0);
		STATUS_SetString("model","status","Failed initializing output tensor");
		return 0;
    }

    infReq = larodCreateInferenceRequest(model, inputTensors, numInputs, outputTensors,numOutputs, &error);
    if (!infReq) {
		STATUS_SetString( "model", "status", "Failed creating inference request" );
        LOG_WARN( "%s: Failed creating inference request: %s\n", __func__, error->msg);
        TFLITE_Close();
		STATUS_SetBool("model","state",0);
		STATUS_SetString("model","status","Failed creating inference request");
		return 0;
    }

	if( !TFLITE_Settings )
		LOG_WARN("%s: CCC TFLITE_Settings is NULL\n",__func__ );


	STATUS_SetNumber( "model", "labels", numberOfLabels );
	STATUS_SetNumber( "model", "inputs", numInputs );
	STATUS_SetNumber( "model", "outputs", numOutputs );

    if (!startFrameFetch(provider)) {
        LOG_WARN( "%s: Unable to start image provider\n",__func__);
        TFLITE_Close();
		STATUS_SetString("model","state","Unable to start image provider");
		return 0;
    }

	STATUS_SetString( "model", "status", "OK" );
	STATUS_SetBool( "model", "state", 1 );	

	HTTP_Node("model",TFLITE_HTTP_Settings);

    return TFLITE_Settings;
}
