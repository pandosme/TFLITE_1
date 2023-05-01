/*
	Fred Juhlin 2023

	Based on https://github.com/AxisCommunications/acap3-examples/tree/main/object-detection
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

#define LOG(fmt, args...)    { syslog(LOG_INFO, fmt, ## args); printf(fmt, ## args);}
#define LOG_WARN(fmt, args...)    { syslog(LOG_WARNING, fmt, ## args); printf(fmt, ## args);}
//#define LOG_TRACE(fmt, args...)    { syslog(LOG_INFO, fmt, ## args); printf(fmt, ## args); }
#define LOG_TRACE(fmt, args...)    {}

// Hardcode to use three image "color" channels (eg. RGB).
const unsigned int CHANNELS = 3;

char* modelFile = "/usr/local/packages/inference/model/model.tflite";
char* labelsFile = "/usr/local/packages/inference/model/labels.txt";
size_t numberOfLabels = 0; // Will be parsed from the labels file

unsigned modelWidth = 224;
unsigned modelHeigth = 224;
unsigned sensorWidth = 1920; //ToDo  read out max resolution
unsigned sensorHeight = 1080;
double confidenceLevel = 60;
unsigned int streamWidth = 0;
unsigned int streamHeight = 0;


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
char** labels = NULL; // This is the array of label strings. The label
					  // entries points into the large labelFileData buffer.
char* labelFileData = NULL; // Buffer holding the complete collection of label strings.

cJSON* TeachableMachine_Settings = 0;
const char* ACAP_PACKAGE = 0;

/**
 * @brief Free up resources held by an array of labels.
 *
 * @param labels An array of label string pointers.
 * @param labelFileBuffer Heap buffer containing the actual string data.
 */
void freeLabels(char** labelsArray, char* labelFileBuffer) {
    free(labelsArray);
    free(labelFileBuffer);
}

/**
 * @brief Reads a file of labels into an array.
 *
 * An array filled by this function should be freed using freeLabels.
 *
 * @param labelsPtr Pointer to a string array.
 * @param labelFileBuffer Pointer to the labels file contents.
 * @param labelsPath String containing the path to the labels file to be read.
 * @param numberOfLabelsPtr Pointer to number which will store number of labels read.
 * @return False if any errors occur, otherwise true.
 */
static bool parseLabels(char*** labelsPtr, char** labelFileBuffer, char *labelsPath, size_t* numberOfLabelsPtr) {
    // We cut off every row at 60 characters.
    const size_t LINE_MAX_LEN = 60;
    bool ret = false;
    char* labelsData = NULL;  // Buffer containing the label file contents.
    char** labelArray = NULL; // Pointers to each line in the labels text.

	LOG_TRACE("%s:\n",__func__);

    struct stat fileStats = {0};
    if (stat(labelsPath, &fileStats) < 0) {
        LOG_WARN( "%s: Unable to get stats for label file %s: %s", __func__, labelsPath, strerror(errno));
        return false;
    }

    // Sanity checking on the file size - we use size_t to keep track of file
    // size and to iterate over the contents. off_t is signed and 32-bit or
    // 64-bit depending on architecture. We just check toward 10 MByte as we
    // will not encounter larger label files and both off_t and size_t should be
    // able to represent 10 megabytes on both 32-bit and 64-bit systems.
    if (fileStats.st_size > (10 * 1024 * 1024)) {
        LOG_WARN( "%s: failed sanity check on labels file size", __func__);
        return false;
    }

    int labelsFd = open(labelsPath, O_RDONLY);
    if (labelsFd < 0) {
        LOG_WARN( "%s: Could not open labels file %s: %s", __func__, labelsPath, strerror(errno));
        return false;
    }

    size_t labelsFileSize = (size_t) fileStats.st_size;
    // Allocate room for a terminating NULL char after the last line.
    labelsData = malloc(labelsFileSize + 50);
    if (labelsData == NULL) {
        LOG_WARN( "%s: Failed allocating labels text buffer: %s", __func__,
               strerror(errno));
        goto end;
    }

    ssize_t numBytesRead = -1;
    size_t totalBytesRead = 0;
    char* fileReadPtr = labelsData;
    while (totalBytesRead < labelsFileSize) {
        numBytesRead =
            read(labelsFd, fileReadPtr, labelsFileSize - totalBytesRead);

        if (numBytesRead < 1) {
            LOG_WARN( "%s: Failed reading from labels file: %s", __func__,
                   strerror(errno));
            goto end;
        }
        totalBytesRead += (size_t) numBytesRead;
        fileReadPtr += numBytesRead;
    }

    // Now count number of lines in the file - check all bytes except the last
    // one in the file.
    size_t numLines = 0;
    for (size_t i = 0; i < (labelsFileSize - 1); i++) {
        if (labelsData[i] == '\n') {
            numLines++;
        }
    }

    // We assume that there is always a line at the end of the file, possibly
    // terminated by newline char. Either way add this line as well to the
    // counter.
    numLines++;

    labelArray = malloc(numLines * sizeof(char*));
    if (!labelArray) {
        LOG_WARN( "%s: Unable to allocate labels array: %s", __func__,
               strerror(errno));
        ret = false;
        goto end;
    }

    size_t labelIdx = 0;
    labelArray[labelIdx] = labelsData;
    labelIdx++;
    for (size_t i = 0; i < labelsFileSize; i++) {
        if (labelsData[i] == '\n') {
            // Register the string start in the list of labels.
            labelArray[labelIdx] = labelsData + i + 1;
            labelIdx++;
            // Replace the newline char with string-ending NULL char.
            labelsData[i] = '\0';
        }
    }

    // If the very last byte in the labels file was a new-line we just
    // replace that with a NULL-char. Refer previous for loop skipping looking
    // for new-line at the end of file.
    if (labelsData[labelsFileSize - 1] == '\n') {
        labelsData[labelsFileSize - 1] = '\0';
    }

    // Make sure we always have a terminating NULL char after the label file
    // contents.
    labelsData[labelsFileSize] = '\0';

    // Now go through the list of strings and cap if strings too long.
    for (size_t i = 0; i < numLines; i++) {
        size_t stringLen = strnlen(labelArray[i], LINE_MAX_LEN);
        if (stringLen >= LINE_MAX_LEN) {
            // Just insert capping NULL terminator to limit the string len.
            *(labelArray[i] + LINE_MAX_LEN + 1) = '\0';
        }
    }

    *labelsPtr = labelArray;
    *numberOfLabelsPtr = numLines;
    *labelFileBuffer = labelsData;

    ret = true;
end:
    if (!ret) {
        freeLabels(labelArray, labelsData);
    }
    close(labelsFd);

    return ret;
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
        LOG_WARN( "%s: Unable to open temp file %s: %s", __func__, fileName, strerror(errno));
        goto error;
    }
    // Allocate enough space in for the fd.
    if (ftruncate(fd, (off_t) fileSize) < 0) {
        LOG_WARN( "%s: Unable to truncate temp file %s: %s", __func__, fileName, strerror(errno));
        goto error;
    }

    // Remove since we don't actually care about writing to the file system.
    if (unlink(fileName)) {
        LOG_WARN( "%s: Unable to unlink from temp file %s: %s", __func__, fileName, strerror(errno));
        goto error;
    }

    // Get an address to fd's memory for this process's memory space.
    void* data = mmap(NULL, fileSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

    if (data == MAP_FAILED) {
        LOG_WARN( "%s: Unable to mmap temp file %s: %s", __func__, fileName, strerror(errno));
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
        LOG_WARN( "%s: Could not connect to larod: %s", __func__, error->msg);
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
				LOG_WARN("No Larod compatible chip found");
				goto error;
			}
		}
    }

    loadedModel = larodLoadModel(conn, larodModelFd, LAROD_ACCESS_PRIVATE, ACAP_PACKAGE, &error);
    if (!loadedModel) {
		STATUS_SetString("model","status", "Unable to load model");
        LOG_WARN( "%s: Unable to load model: %s", __func__, error->msg);
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
TeachableMachine_Inference() {

	struct timeval startTs, endTs;
	unsigned int elapsedMs = 0;

	if( inferenceRunning )
		return 0;

	if( !provider) {
		STATUS_SetString("model","state","No image provider");
		return 0;
	}

	// Get latest frame from image pipeline.
	VdoBuffer* buf = getLastFrameBlocking(provider);
	if (!buf) {
		LOG_WARN( "%s: No image avaialable\n", __func__ );
		STATUS_SetString("model","state","No image provider");
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
		LOG_WARN( "%s: Unable to run inference on model %s: %s (%d)\n", __func__, modelFile, error->msg, error->code);
		larodClearError(&error);		
		inferenceRunning = 0;
		return 0;
	}

	inferenceRunning = 0;

	gettimeofday(&endTs, NULL);

	elapsedMs = (unsigned int) (((endTs.tv_sec - startTs.tv_sec) * 1000) +
								((endTs.tv_usec - startTs.tv_usec) / 1000));

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
		if( score >=  confidenceLevel ) {
			cJSON* item = cJSON_CreateObject();
			cJSON_AddStringToObject( item,"label",labels[i] );
			cJSON_AddNumberToObject( item,"score", (int)score );
			cJSON_AddItemToArray(list,item);
		}
	}
	
	returnFrame(provider, buf);
	LOG_TRACE("%s: Exit\n",__func__);
	return payload;
}

static void
TeachableMachine_HTTP_Settings(const HTTP_Response response,const HTTP_Request request) {
	
	const char* json = HTTP_Request_Param( request, "json");
	if( !json )
		json = HTTP_Request_Param( request, "set");
	if( !json ) {
		HTTP_Respond_JSON( response, TeachableMachine_Settings );
		return;
	}

	cJSON *params = cJSON_Parse(json);
	if(!params) {
		HTTP_Respond_Error( response, 400, "Invalid JSON data");
		return;
	}
	LOG_TRACE("%s: %s\n",__func__,json);
	cJSON* param = params->child;
	while(param) {
		if( cJSON_GetObjectItem(TeachableMachine_Settings,param->string ) )
			cJSON_ReplaceItemInObject(TeachableMachine_Settings,param->string,cJSON_Duplicate(param,1) );
		param = param->next;
	}
	cJSON_Delete(params);

	confidenceLevel = cJSON_GetObjectItem(TeachableMachine_Settings,"confidence")?cJSON_GetObjectItem(TeachableMachine_Settings,"confidence")->valuedouble:60.0;
	
	FILE_Write( "localdata/TeachableMachine.json", TeachableMachine_Settings);
	HTTP_Respond_Text( response, "OK" );
}

void 
TeachableMachine_Close() {
	
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

    if (labels) {
        freeLabels(labels, labelFileData);
    }

	STATUS_SetString( "model", "status", "Not avaialble" );
	STATUS_SetBool( "model", "state", 0 );	
	STATUS_SetString( "model", "acrhitecture", "Undefined" );	

}

cJSON*
TeachableMachine( const char* package ) {
	LOG_TRACE("%s: \n",__func__);
	ACAP_PACKAGE = package;
	STATUS_SetString( "model", "status", "Initializing" );
	STATUS_SetBool( "model", "state", 0 );	
	STATUS_SetString( "model", "architecture", "Undefined" );	

	cJSON* TeachableMachine_Settings = FILE_Read( "html/config/TeachableMachine.json" );
	if(!TeachableMachine_Settings)
		TeachableMachine_Settings = cJSON_CreateObject();

	cJSON* savedSettings = FILE_Read( "localdata/settings.json" );
	if( savedSettings ) {
		cJSON* prop = savedSettings->child;
		while(prop) {
			if( cJSON_GetObjectItem(TeachableMachine_Settings,prop->string ) )
				cJSON_ReplaceItemInObject(TeachableMachine_Settings,prop->string,cJSON_Duplicate(prop,1) );
			prop = prop->next;
		}
		cJSON_Delete(savedSettings);
	}

	confidenceLevel = cJSON_GetObjectItem(TeachableMachine_Settings,"confidence")?cJSON_GetObjectItem(TeachableMachine_Settings,"confidence")->valuedouble:60.0;

    if (!chooseStreamResolution(modelWidth, modelHeigth, &streamWidth,&streamHeight)) {
        LOG_WARN( "%s: Failed choosing stream resolution", __func__);
        TeachableMachine_Close();
		STATUS_SetString("model","state","No valid stream resolutions");
		return 0;
    }
	
    if (!labelsFile) {
        LOG_WARN( "%s: Missing labels file\n",__func__);
		TeachableMachine_Close();
		STATUS_SetString( "model", "status", "Missing labels file" );
		return 0;
	}
	
    if (!parseLabels(&labels, &labelFileData, labelsFile, &numberOfLabels)) {
        LOG_WARN( "%s: Failed creating parsing labels file\n",__func__);
		TeachableMachine_Close();
		STATUS_SetString("model","state","Invalid labels file");
		return 0;
    }

    provider = createImgProvider(streamWidth, streamHeight, 2, VDO_FORMAT_YUV);
    if (!provider) {
		LOG_WARN( "%s: Failed to create ImgProvider", __func__);
        TeachableMachine_Close();
		STATUS_SetString("model","state","Failed to create image provider");
		return 0;
    }

    larodModelFd = open(modelFile, O_RDONLY);
    if (larodModelFd < 0) {
        LOG_WARN( "%s: Unable to open model file %s: %s", __func__,modelFile,  strerror(errno));
        TeachableMachine_Close();
		STATUS_SetString("model","state","Model file does not exist");
		return 0;
    }
    if (!setupLarod(larodModelFd, &conn, &model)) {
        TeachableMachine_Close();
		STATUS_SetString("model","state","Failed setting up architecture");
		return 0;
    }

    // Allocate space for input tensor
    if (!createAndMapTmpFile(CONV_INP_FILE_PATTERN, modelWidth * modelHeigth * CHANNELS, &larodInputAddr, &larodInputFd)) {
		STATUS_SetString( "model", "status", "Allocation failed" );
        TeachableMachine_Close();
		STATUS_SetString("model","state","Input data allocation failed");
		return 0;
    }
    // Allocate space for output tensor 1 (Locations)
    if (!createAndMapTmpFile(CONV_OUT1_FILE_PATTERN, numberOfLabels,  &larodOutput1Addr, &larodOutput1Fd)) {
		STATUS_SetString( "model", "status", "Allocation failed" );
        TeachableMachine_Close();
		STATUS_SetString("model","state","Output data allocation failed");
		return 0;
    }
    inputTensors = larodCreateModelInputs(model, &numInputs, &error);
    if (!inputTensors) {
		STATUS_SetString( "model", "status", "Failed retrieving input tensors" );
        LOG_WARN( "Failed retrieving input tensors: %s", error->msg);
        TeachableMachine_Close();
		STATUS_SetString("model","state","Failed initializing input tensor");
		return 0;
    }
    if (!larodSetTensorFd(inputTensors[0], larodInputFd, &error)) {
		STATUS_SetString( "model", "status", "Failed setting input tensor" );
        LOG_WARN( "%s: Failed setting input tensor fd: %s\n", __func__,error->msg);
        TeachableMachine_Close();
		STATUS_SetString("model","state","Failed initializing input tensor");
		return 0;
    }
    outputTensors = larodCreateModelOutputs(model, &numOutputs, &error);
    if (!outputTensors) {
		STATUS_SetString( "model", "status", "Failed retrieving output tensors" );
        LOG_WARN( "%s: Failed retrieving output tensors: %s\n", __func__, error->msg);
        TeachableMachine_Close();
		STATUS_SetString("model","state","Failed initializing output tensor");
		return 0;
    }

    if (!larodSetTensorFd(outputTensors[0], larodOutput1Fd, &error)) {
		STATUS_SetString( "model", "status", "Failed setting output tensor" );
        LOG_WARN( "%s: Failed setting output tensor fd: %s\n", __func__, error->msg);
        TeachableMachine_Close();
		STATUS_SetString("model","state","Failed initializing output tensor");
		return 0;
    }

    infReq = larodCreateInferenceRequest(model, inputTensors, numInputs, outputTensors,numOutputs, &error);
    if (!infReq) {
		STATUS_SetString( "model", "status", "Failed creating inference request" );
        LOG_WARN( "%s: Failed creating inference request: %s\n", __func__, error->msg);
        TeachableMachine_Close();
		STATUS_SetString("model","state","Failed creating inference request");
		return 0;
    }


	STATUS_SetNumber( "model", "labels", numberOfLabels );
	STATUS_SetNumber( "model", "inputs", numInputs );
	STATUS_SetNumber( "model", "outputs", numOutputs );

    if (!startFrameFetch(provider)) {
        LOG_WARN( "%s: Unable to start image provider\n",__func__);
        TeachableMachine_Close();
		STATUS_SetString("model","state","Unable to start image provider");
		return 0;
    }

	STATUS_SetString( "model", "status", "OK" );
	STATUS_SetBool( "model", "state", 1 );	

	HTTP_Node("TeachableMachine",TeachableMachine_HTTP_Settings);

    return TeachableMachine_Settings;
}
