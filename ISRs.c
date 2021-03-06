///////////////////////////////////////////////////////////////////////
// Filename: ISRs.c
//
// Synopsis: Interrupt service routine for codec data transmit/receive
//
///////////////////////////////////////////////////////////////////////

#include "DSP_Config.h" 

// Data is received as 2 16-bit words (left/right) packed into one
// 32-bit word.  The union allows the data to be accessed as a single 
// entity when transferring to and from the serial port, but still be 
// able to manipulate the left and right channels independently.

#define LEFT  0
#define RIGHT 1

volatile union {
	Uint32 UINT;
	Int16 Channel[2];
} CodecDataIn, CodecDataOut;

struct cmpx
    {
    float real;
    float imag;
    };
typedef struct cmpx COMPLEX;

extern int startflag;
extern int kk;
extern int M;

extern short X[512];

interrupt void Codec_ISR()
///////////////////////////////////////////////////////////////////////
// Purpose:   Codec interface interrupt service routine  
//
// Input:     None
//
// Returns:   Nothing
//
// Calls:     CheckForOverrun, ReadCodecData, WriteCodecData
//
// Notes:     None
///////////////////////////////////////////////////////////////////////
{                    
 	if(CheckForOverrun())
		return;

  	CodecDataIn.UINT = ReadCodecData();

	if(kk>M-1){
         /* (1). Initialize index kk                                            */
		kk=0;
         /* (2). Change startflag to start processing in while loop in main()   */
		startflag = 1;
	}

	if(!startflag){
         /* (1). Put a new data to the buffer X    */
		X[kk] = CodecDataIn.Channel[0];
         /* (2). Update index kk                   */
		kk++;
       }

	WriteCodecData(CodecDataIn.UINT);
}
