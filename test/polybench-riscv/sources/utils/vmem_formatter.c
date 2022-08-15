#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define WORDS_PER_LINE  1
#define BYTES_PER_WORD  4
#define BIT_PER_ADDR    8

#define TRUE            1
#define FALSE           0

int main(int argc, char *argv[]) {
  FILE *fin = NULL;
  FILE *fout = NULL;

  if(argc < 3) {
    fprintf(stderr, "Error. Insufficient options.\n");
    fprintf(stderr, "Usage: ./vmem_formatter <source_file> <dest_file>\n");
    fprintf(stderr, "\t<source_file>:\t.vmem source file to format.\n");
    fprintf(stderr, "\t<dest_file>:\t.vmem destination file.\n");

    return 1;
  }

  fin = fopen(argv[1], "r");        // Open source file with read permissions
  assert(fin && "Unable to open source vmem file");
  
  fout = fopen(argv[2], "w");       // Open destination file with write permissions
  assert(fout && "Unable to open destination vmem file");

  // Temp address to turn it into an integer. Discard '@' character from addr
  char tmp_addr[BIT_PER_ADDR+1] = {'0', '0', '0', '0', '0', '0', '0', '0', '\0'};

  int addr_int = 0;                 // Read address converted from string to int type
  char buff[BIT_PER_ADDR+2];        // Read string buffer. BIT_PER_ADDR+2 is the maximum string length
                                    // when reading
  int word_byte_index = 0;          // Byte index within a 32-bit word. MSByte is 0, LSByte is 3
  int word_index = 0;               // Current word in the current line
  int prev_word_is_byte = FALSE;
  int first_addr = TRUE;

  // ***** Parser logic *****
  while(fscanf(fin, "%s", buff) != EOF) {
    // 1. Check if buff contains an address
    if(buff[0] == '@') {
      if(!first_addr) {
        fprintf(fout, "\n");
      }
      
      // Convert address string to int
      // Do not consider first character '@' and string terminator
      for(int i = 1; i < BIT_PER_ADDR+1; i++) {
        tmp_addr[i] = buff[i];
      }
      sscanf(tmp_addr, "%x", &addr_int);
      
      addr_int = addr_int >> 2;     // Shift left by 2 (word addressing)
      
      fprintf(fout, "@%.8X ", addr_int);
      first_addr = FALSE;
      prev_word_is_byte = FALSE;
      word_index = 0;
      word_byte_index = 0;
    }
    
    // 2. Check if buff contains a byte
    else {
      // Create new line with address and 32-bit word
      if(word_byte_index > BYTES_PER_WORD - 1) 
	  {
      	if(word_index >= WORDS_PER_LINE - 1) 
		{
          fprintf(fout, "\n");
          word_byte_index = 0;
          word_index = 0;
        }
        else {
          fprintf(fout, " ");
          word_byte_index = 0;
          word_index++;
        }
      }

      // Print @address
      if(word_byte_index == 0 && word_index == 0 && prev_word_is_byte) {
        // Increment start_addr_int and turn it back into a string
        addr_int += WORDS_PER_LINE;
        
        // Create new line with address and 32-bit words
        fprintf(fout, "@%.8X ", addr_int);
      }

      fprintf(fout, "%s", buff);
      word_byte_index++;
      prev_word_is_byte = TRUE;
    }
  }

  fprintf(fout, "\n");

  // Close open files
  fclose(fin);
  fclose(fout);

  return 0;
}
