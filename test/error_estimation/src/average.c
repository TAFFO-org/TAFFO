#include <stdio.h>
#include <stdlib.h>


int main(int argc, char * argv[])
{
  if (argc < 3) {
    printf("specify file name and number of values to read from file");
  }
  long n = strtol(argv[1], NULL, 10);
  char* file_name = argv[2];
  FILE *data_file;
  float num;
  float sum = 0;

  data_file = fopen(file_name, "r");

  if (NULL == data_file)
  {
    perror("opening data file");
    return (-1);
  }

  for (long i = 0; i < n; i++) {
    if (EOF == fscanf(data_file, "%f\n", &num)) {
      break ;
    }
    sum += num;
  }
  float average = sum / n;
  printf("%f\n", average);

  fclose(data_file);
  return (0);
}
