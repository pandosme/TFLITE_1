/*------------------------------------------------------------------
 *  Fred Juhlin (2023)
 *------------------------------------------------------------------*/
 
#ifndef _TeachableMachine_H_
#define _TeachableMachine_H_

#ifdef  __cplusplus
extern "C" {
#endif

cJSON*  TeachableMachine( const char *package );  //Returns settings
void 	TeachableMachine_Close();
cJSON*	TeachableMachine_Inference();  //Note that response needs to be deleted with cHSON_Detete

#ifdef  __cplusplus
}
#endif

#endif
