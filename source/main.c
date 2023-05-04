/*
	Fred Juhlin 2023
	
	TFLITE inferance can be initiated by
	1. HTTP request /tflite/inference
	2. Timer

	inference object
	{
		"device": "SERIALNUMBER",
		"timstamp": EPOCH ms resolution
		"duration": number of milli seconds the inference took,
		"list": [
			{ "lable": "Some label","score": 0 - 100},
			{...}
		]
	}
	
*/

#include <stdlib.h>
#include <stdio.h>
#include <glib.h>
#include <glib/gi18n.h>
#include <string.h>
#include <syslog.h>

#include "APP.h"
#include "HTTP.h"
#include "TFLITE_1.h"

#define LOG(fmt, args...)    { syslog(LOG_INFO, fmt, ## args); printf(fmt, ## args);}
#define LOG_WARN(fmt, args...)    { syslog(LOG_WARNING, fmt, ## args); printf(fmt, ## args);}
//#define LOG_TRACE(fmt, args...)    { syslog(LOG_INFO, fmt, ## args); printf(fmt, ## args); }
#define LOG_TRACE(fmt, args...)    {}

#define APP_PACKAGE	"tflite"

volatile sig_atomic_t stopRunning = 0;
cJSON* tfliteModel = 0;


static void
Inference_HTTP(const HTTP_Response response,const HTTP_Request request) {
	
	if( !tfliteModel ) {
		HTTP_Respond_Error( response, 500, "Model is not initialized" );
		return;
	}

	cJSON* inference = TFLITE_Inference();
	if(!inference) {
		HTTP_Respond_Error( response, 500, "Inference failed" );
		return;
	}
	
	HTTP_Respond_JSON( response, inference );
	cJSON_Delete(inference);
}

static gboolean
Inference_Timer() {

	cJSON* inference = TFLITE_Inference();

	if(!inference)
		return TRUE;

	cJSON* list = cJSON_GetObjectItem(inference,"list");
	if(!list) {
		cJSON_Delete(inference);
		return TRUE;
	}
	
	int numberOfDetections = cJSON_GetArraySize( list );
	if( numberOfDetections == 0 ) {
		cJSON_Delete(inference);
		return TRUE;
	}

	LOG("%d detections\n", numberOfDetections);

	//Iterate over detections
	cJSON* detection = list->child;
	while( detection ) {
		const char* label = cJSON_GetObjectItem(detection,"label")?cJSON_GetObjectItem(detection,"label")->valuestring:"Undefined";
		int score = cJSON_GetObjectItem(detection,"score")?cJSON_GetObjectItem(detection,"score")->valueint:0;
		LOG("%s %d\n", label, score );
		detection = detection->next;
	}
	return TRUE;
}


void sigintHandler(int sig) {
    if (stopRunning) {
        LOG_TRACE( "Interrupted again, exiting immediately without clean up.");
        exit(EXIT_FAILURE);
    }

    LOG_TRACE( "Interrupted, starting graceful termination of app. Another interrupt signal will cause a forced exit.");
    stopRunning = 1;
}

int
main() {
	openlog(APP_PACKAGE, LOG_PID|LOG_CONS, LOG_USER);
	signal(SIGINT, sigintHandler);

	GMainLoop *loop;

	APP( APP_PACKAGE, NULL );
	
	tfliteModel = TFLITE(APP_PACKAGE);
	if( tfliteModel ) {
		APP_Register("model",tfliteModel);
		HTTP_Node("inference",Inference_HTTP);

		//Run Inference every 5 seconds
		g_timeout_add_seconds( 5, Inference_Timer, NULL );

		loop = g_main_loop_new(NULL, FALSE);
		g_main_loop_run(loop);
	} else {
		LOG_WARN("Stopped.  Unable to initialize model\n");
	}

	TFLITE_Close();
}
