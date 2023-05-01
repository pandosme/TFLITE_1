/*
	Fred Juhlin 2023
*/

#include <stdlib.h>
#include <stdio.h>
#include <glib.h>
#include <glib/gi18n.h>
#include <string.h>
#include <syslog.h>

#include "APP.h"
#include "HTTP.h"
#include "TeachableMachine.h"

#define LOG(fmt, args...)    { syslog(LOG_INFO, fmt, ## args); printf(fmt, ## args);}
#define LOG_WARN(fmt, args...)    { syslog(LOG_WARNING, fmt, ## args); printf(fmt, ## args);}
//#define LOG_TRACE(fmt, args...)    { syslog(LOG_INFO, fmt, ## args); printf(fmt, ## args); }
#define LOG_TRACE(fmt, args...)    {}

#define APP_PACKAGE	"inference"

volatile sig_atomic_t stopRunning = 0;
cJSON* modelSettings = 0;


static void
run_HTTP(const HTTP_Response response,const HTTP_Request request) {
	
	if( !modelSettings ) {
		HTTP_Respond_Error( response, 500, "Teachable Machine is not initialized" );
		return;
	}

	cJSON* inference = TeachableMachine_Inference();
	if(!inference) {
		HTTP_Respond_Error( response, 500, "Inference failed" );
		return;
	}
	
	HTTP_Respond_JSON( response, inference );
	cJSON_Delete(inference);
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
	modelSettings = TeachableMachine(APP_PACKAGE);
	if( modelSettings )
		APP_Register("TeachableMachine",modelSettings);
	else
		LOG_WARN("Unable to initialize model\n");

	HTTP_Node("run",run_HTTP);

	loop = g_main_loop_new(NULL, FALSE);
	g_main_loop_run(loop);
	
	TeachableMachine_Close();
}
