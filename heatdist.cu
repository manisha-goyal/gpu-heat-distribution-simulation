/* 
 * This file contains the code for doing the heat distribution problem. 
 * You do not need to modify anything except starting  gpu_heat_dist() at the bottom
 * of this file.
 * In gpu_heat_dist() you can organize your data structure and the call to your
 * kernel(s), memory allocation, data movement, etc. 
 * 
 */

#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h> 

/* To index element (i,j) of a 2D square array of dimension NxN stored as 1D 
   index(i, j, N) means access element at row i, column j, and N is the dimension which is NxN */
#define index(i, j, N)  ((i)*(N)) + (j)

/*****************************************************************/

// Function declarations: Feel free to add any functions you want.
void  seq_heat_dist(float *, unsigned int, unsigned int);
void  gpu_heat_dist(float *, unsigned int, unsigned int);
__global__ void heat_kernel(float *playground, float *temp, unsigned int N);

/*****************************************************************/
/**** Do NOT CHANGE ANYTHING in main() function ******/

int main(int argc, char * argv[])
{
  unsigned int N; /* Dimention of NxN matrix */
  int type_of_device = 0; // CPU or GPU
  int iterations = 0;
  int i;
  
  /* The 2D array of points will be treated as 1D array of NxN elements */
  float * playground; 
  
  // to measure time taken by a specific part of the code 
  double time_taken;
  clock_t start, end;
  
  if(argc != 4)
  {
    fprintf(stderr, "usage: heatdist num  iterations  who\n");
    fprintf(stderr, "num = dimension of the square matrix (50 and up)\n");
    fprintf(stderr, "iterations = number of iterations till stopping (1 and up)\n");
    fprintf(stderr, "who = 0: sequential code on CPU, 1: GPU version\n");
    exit(1);
  }
  
  type_of_device = atoi(argv[3]);
  N = (unsigned int) atoi(argv[1]);
  iterations = (unsigned int) atoi(argv[2]);
 
  
  /* Dynamically allocate NxN array of floats */
  playground = (float *)calloc(N*N, sizeof(float));
  if( !playground )
  {
   fprintf(stderr, " Cannot allocate the %u x %u array\n", N, N);
   exit(1);
  }
  
  /* Initialize it: calloc already initalized everything to 0 */
  // Edge elements  initialization
  for(i = 0; i < N; i++)
    playground[index(0,i,N)] = 100;
  for(i = 0; i < N; i++)
    playground[index(N-1,i,N)] = 150;
  for(i = 1; i < N-1; i++)
    playground[index(i,0,N)] = 80;
  for(i = 1; i < N-1; i++)
    playground[index(i,N-1,N)] = 80;
  

  switch(type_of_device)
  {
	case 0: printf("CPU sequential version:\n");
			start = clock();
			seq_heat_dist(playground, N, iterations);
			end = clock();
			break;
		
	case 1: printf("GPU version:\n");
			start = clock();
			gpu_heat_dist(playground, N, iterations); 
			cudaDeviceSynchronize();
			end = clock();  
			break;
			
	default: printf("Invalid device type\n");
			 exit(1);
  }
  
  time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
  
  printf("Time taken = %lf\n", time_taken);
  
  free(playground);
  
  return 0;

}


/*****************  The CPU sequential version (DO NOT CHANGE THAT) **************/
void  seq_heat_dist(float * playground, unsigned int N, unsigned int iterations)
{
  // Loop indices
  int i, j, k;
  int upper = N-1;
  
  // number of bytes to be copied between array temp and array playground
  unsigned int num_bytes = 0;
  
  float * temp; 
  /* Dynamically allocate another array for temp values */
  /* Dynamically allocate NxN array of floats */
  temp = (float *)calloc(N*N, sizeof(float));
  if( !temp )
  {
   fprintf(stderr, " Cannot allocate temp %u x %u array\n", N, N);
   exit(1);
  }
  
  num_bytes = N*N*sizeof(float);
  
  /* Copy initial array in temp */
  memcpy((void *)temp, (void *) playground, num_bytes);
  
  for( k = 0; k < iterations; k++)
  {
    /* Calculate new values and store them in temp */
    for(i = 1; i < upper; i++)
      for(j = 1; j < upper; j++)
	temp[index(i,j,N)] = (playground[index(i-1,j,N)] + 
	                      playground[index(i+1,j,N)] + 
			      playground[index(i,j-1,N)] + 
			      playground[index(i,j+1,N)])/4.0;
  
			      
   			      
    /* Move new values into old values */ 
    memcpy((void *)playground, (void *) temp, num_bytes);
  }
  
}

/***************** The GPU version: Write your code here *********************/
/* This function can call one or more kernels if you want ********************/
void  gpu_heat_dist(float * playground, unsigned int N, unsigned int iterations)
{
  
  /* Here you have to write any cuda dynamic allocations, any communications between device and host, any number of kernel
     calls, etc. */

  float *playground_d, *temp_d;
  size_t size = N * N * sizeof(float);

  //Allocate the playground and temp playground in the device
  if (cudaMalloc((void**)&playground_d, size) != cudaSuccess)
	{
		printf("Error allocating playground of %dx%d elements on device\n", N, N);
		exit(1);
	}
  if (cudaMalloc((void**)&temp_d, size) != cudaSuccess)
	{
		printf("Error allocating temp playground of %dx%d elements on device\n", N, N);
		exit(1);
	}

  //Copy playground to the device
	if (cudaMemcpy(playground_d, playground, size, cudaMemcpyHostToDevice) != cudaSuccess) {
    printf("Error copying playground from host to device\n");
    exit(1);
  }
  if (cudaMemcpy(temp_d, playground, size, cudaMemcpyHostToDevice) != cudaSuccess) {
    printf("Error copying playground from host to device\n");
    exit(1);
  }

  // Define a 2D block
  dim3 block(16, 16);
  // Define a 2D grid of blocks to fully cover an N x N data grid
  dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

  // Run the kernel for the specified number of iterations
  for (int k = 0; k < iterations; k++) {
      heat_kernel<<<grid, block>>>(playground_d, temp_d, N);
      cudaDeviceSynchronize();

      // Copy playground data for next iteration
      float* temp = playground_d;
      playground_d = temp_d;
      temp_d = temp;
  }

  // Copy the result back to the host
  if (cudaMemcpy(playground, playground_d, size, cudaMemcpyDeviceToHost) != cudaSuccess) {
    printf("Error copying result from device to host\n");
    exit(1);
  }

  //Free the playground and temp playground in the device
  cudaFree(playground_d);
  cudaFree(temp_d);
}

/* Kernel code */
__global__ void heat_kernel(float *playground, float *temp, unsigned int N) {
  int threadId_x = blockIdx.x * blockDim.x + threadIdx.x;
  int threadId_y = blockIdx.y * blockDim.y + threadIdx.y;
  int offset_x = blockDim.x * gridDim.x;
  int offset_y = blockDim.y * gridDim.y;

  // Traverse the grid with strides in both x and y directions
  for (int i = threadId_x; i < N; i += offset_x) {
    for(int j = threadId_y; j < N; j += offset_y)
      // Only update interior points
      if (i > 0 && i < N - 1 && j > 0 && j < N - 1) {
        // Calculate the average temperature of neighboring points
        temp[index(i, j, N)] = (playground[index(i - 1, j, N)] +
                                playground[index(i + 1, j, N)] +
                                playground[index(i, j - 1, N)] +
                                playground[index(i, j + 1, N)]) / 4.0;
      }
  }
}