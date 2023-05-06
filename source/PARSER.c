/*------------------------------------------------------------------
 *  Fred Juhlin (2023)
 *------------------------------------------------------------------*/

#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <syslog.h>
#include <float.h>
#include <limits.h>
#include <ctype.h>
#include "PARSER.h"

#define LOG(fmt, args...)    { syslog(LOG_INFO, fmt, ## args); printf(fmt, ## args);}
#define LOG_WARN(fmt, args...)    { syslog(LOG_WARNING, fmt, ## args); printf(fmt, ## args);}
//#define LOG_TRACE(fmt, args...)    { syslog(LOG_INFO, fmt, ## args); printf(fmt, ## args); }
#define LOG_TRACE(fmt, args...)    {}


/**
 * @brief Split a string based on a delimiter
 *
 * @param labels An array of label string pointers.
 * @param labelFileBuffer Heap buffer containing the actual string data.
 */
char**
PARSER_StringSplit2Array( char* aString,  char aDelimiter) {
    char** result    = 0;
    size_t count     = 0;
    const char* tmp  = aString;
    const char* last_comma = 0;
    char delim[2];
    delim[0] = aDelimiter;
    delim[1] = 0;
    /* Count how many elements will be extracted. */
    while (*tmp)
    {
        if (aDelimiter == *tmp)
        {
            count++;
            last_comma = tmp;
        }
        tmp++;
    }

    /* Add space for trailing token. */
    count += last_comma < (aString + strlen(aString) - 1);

    /* Add space for terminating null string so caller
       knows where the list of returned strings ends. */
    count++;
    result = malloc(sizeof(char*) * count);
    if (result) {
        size_t idx  = 0;
        char* token = strtok(aString, delim);
        while (token) {
//            assert(idx < count);
            *(result + idx++) = strdup(token);
            token = strtok(0, delim);
        }
//        assert(idx == count - 1);
        *(result + idx) = 0;
    }
    return result;
}

cJSON*
PARSER_SplitToJSON( char* aString,  char aDelimiter ) {
	cJSON* list = cJSON_CreateArray();
	char **stringArray;
	stringArray = PARSER_StringSplit2Array( aString, aDelimiter );
	if( !stringArray ) {
		LOG_WARN("%s: Unable to parse string\n",__func__);
		return list;
	}

	int i;
	for (i = 0; *(stringArray + i); i++) {
		if( strlen(*(stringArray + i)) > 0 )
			cJSON_AddItemToArray(list, cJSON_CreateString(*(stringArray + i)));
		free(*(stringArray + i));
	}
	free(stringArray);
	return list;
}