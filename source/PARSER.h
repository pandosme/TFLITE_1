/*------------------------------------------------------------------
 *  Fred Juhlin (2023)
 *------------------------------------------------------------------*/

#ifndef _PARSER_H_
#define _PARSER_H_

#include "cJSON.h"

#ifdef __cplusplus
extern "C"
{
#endif

char** PARSER_StringSplit2Array( char* someString,  char token);
cJSON* PARSER_SplitToJSON( char* someString,  char token);

#ifdef __cplusplus
}
#endif

#endif
