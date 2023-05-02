/*------------------------------------------------------------------
 *  Fred Juhlin (2023)
 *------------------------------------------------------------------*/
 
#ifndef _TFLITE_H_
#define _TFLITE_H_


#ifdef  __cplusplus
extern "C" {
#endif

cJSON*  TFLITE( const char *package );  //Returns settings
void 	TFLITE_Close();
cJSON*	TFLITE_Inference();  //Note that response needs to be deleted with cHSON_Detete

#ifdef  __cplusplus
}
#endif

#endif
