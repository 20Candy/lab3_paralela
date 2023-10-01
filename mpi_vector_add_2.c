/* File:     mpi_vector_add.c
 *
 * Purpose:  Implement parallel vector addition using a block
 *           distribution of the vectors.  This version also
 *           illustrates the use of MPI_Scatter and MPI_Gather.
 *
 * Compile:  mpicc -g -Wall -o mpi_vector_add mpi_vector_add.c
 * Run:      mpiexec -n <comm_sz> ./mpi_vector_add
 *
 * Input:    The order of the vectors, n, and the vectors x and y
 * Output:   The point product of x and y, the mult. x*scalar and y*scalar
 *
 * Notes:     
 * 1.  The order of the vectors, n, should be evenly divisible
 *     by comm_sz
 * 2.  DEBUG compile flag.    
 * 3.  This program does fairly extensive error checking.  When
 *     an error is detected, a message is printed and the processes
 *     quit.  Errors detected are incorrect values of the vector
 *     order (negative or not evenly divisible by comm_sz), and
 *     malloc failures.
 *
 * IPP:  Section 3.4.6 (pp. 109 and ff.)
 */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

void Check_for_error(int local_ok, char fname[], char message[], 
      MPI_Comm comm);
void Read_n(int* n_p, int* local_n_p, int* escalar, int my_rank, int comm_sz, 
      MPI_Comm comm);
void Allocate_vectors(double** local_x_pp, double** local_y_pp,
      double** local_z_pp, double **local_a_pp, int local_n, MPI_Comm comm);
void Read_vector(double local_a[], int local_n, int n, char vec_name[], 
      int my_rank, MPI_Comm comm);
void Print_vector(double local_b[], int local_n, int n, char title[], 
      int my_rank, MPI_Comm comm);
void Parallel_vector_sum(double local_x[], double local_y[], 
      double local_z[], int local_n);

/* Prototipos de funciones adicionales */
double Parallel_vector_dot_product(double local_x[], double local_y[], int local_n, MPI_Comm comm);
void Parallel_scalar_multiply(double local_x[], double local_z[], double scalar, int local_n);



/*-------------------------------------------------------------------*/
int main(void) {
   int n, local_n, escalar;                     // n = tamaño de los vectores, local_n = tamaño de los vectores locales, escalar = valor del escalar
   int comm_sz, my_rank;                        // comm_sz = número de procesos, my_rank = rango del proceso
   MPI_Comm comm;     

   double start, elapsed;

   double *local_x, *local_y, *local_z, *local_a;     // vectores locales
   double *x, *y, *z, *a;                             // VECTORES GLOBALES. Se utiliza para combinar los vectores locales.

   MPI_Init(NULL, NULL);                              // Inicializa el ambiente de MPI
   comm = MPI_COMM_WORLD;
   MPI_Comm_size(comm, &comm_sz);                     // Obtiene el número de procesos
   MPI_Comm_rank(comm, &my_rank);                     // Obtiene el rango del proceso

   //n = 100000;                                      // Ejercicio 2: Crear dos vectores de al menos 100,000 elementos generados de forma aleatoria
   Read_n(&n, &local_n, &escalar, my_rank, comm_sz, comm);                        // Se lee el tamaño de los vectores (n) y el valor del escalar y se distribuye entre los procesos


   if (my_rank == 0) start = MPI_Wtime();


   Allocate_vectors(&local_x, &local_y, &local_z, &local_a, local_n, comm);         // Se le asigna memoria a los vectores locales
   Allocate_vectors(&x, &y, &z, &a, n, comm);                                       // Se le asigna memoria a los vectores globales

   Read_vector(local_x, local_n, n, "x", my_rank, comm);                // Se llenan los vectores con números aleatorios
   Read_vector(local_y, local_n, n, "y", my_rank, comm);                // Se llenan los vectores con números aleatorios
   
   // Se calcula el producto punto de los vectores locales
   double local_dot_product = Parallel_vector_dot_product(local_x, local_y, local_n, comm);

   // Multiplicar vectores por un escalar
   Parallel_scalar_multiply(local_x, local_z, escalar, local_n);
   Parallel_scalar_multiply(local_y, local_a, escalar, local_n);

   // Se suman los vectores locales en uno solo
   //Parallel_vector_sum(local_x, local_y, local_z, local_n); 

   if (my_rank == 0) elapsed = MPI_Wtime() - start;


   // SE COMBINAN LOS VECTORES LOCALES EN UNO SOLO DESPUES DE HABER SUMADO
   MPI_Gather(local_x, local_n, MPI_DOUBLE, x, local_n, MPI_DOUBLE, 0, comm);
   MPI_Gather(local_y, local_n, MPI_DOUBLE, y, local_n, MPI_DOUBLE, 0, comm);
   MPI_Gather(local_z, local_n, MPI_DOUBLE, z, local_n, MPI_DOUBLE, 0, comm);
   MPI_Gather(local_a, local_n, MPI_DOUBLE, a, local_n, MPI_DOUBLE, 0, comm);
   
   if (my_rank == 0){

      printf("Size of vectors: %d\n", n); // Se imprime el tamaño de los vectores (n)

      printf("\nProducto punto = %f\n", local_dot_product);

      printf("\nVector X * %d = \n", escalar);            
      printf("\nX\t\t|\tZ (Resultado)\n\n");            // Se imprimen los vectores

      for (int i = 0; i < n; i++) {
            printf("%f\t|\t%f\n", x[i], z[i]);
      }


      printf("\nVector Y * %d = \n", escalar);
      printf("\nY\t\t|\tA (Resultado)\n\n");            // Se imprimen los vectores

      for (int i = 0; i < n; i++) {
            printf("%f\t|\t%f\n", y[i], a[i]);
      }

      printf("\nTime elapsed: %f seconds\n", elapsed);

   }

   free(x);
   free(y);
   free(z);
   free(a);
   
   free(local_x);
   free(local_y);
   free(local_z);
   free(local_a);

   MPI_Finalize();

   return 0;
}  /* main */

/*-------------------------------------------------------------------
 * Function:  Check_for_error
 * Purpose:   Check whether any process has found an error.  If so,
 *            print message and terminate all processes.  Otherwise,
 *            continue execution.
 * In args:   local_ok:  1 if calling process has found an error, 0
 *               otherwise
 *            fname:     name of function calling Check_for_error
 *            message:   message to print if there's an error
 *            comm:      communicator containing processes calling
 *                       Check_for_error:  should be MPI_COMM_WORLD.
 *
 * Note:
 *    The communicator containing the processes calling Check_for_error
 *    should be MPI_COMM_WORLD.
 */
void Check_for_error(
      int       local_ok   /* in */, 
      char      fname[]    /* in */,
      char      message[]  /* in */, 
      MPI_Comm  comm       /* in */) {
   int ok;

   MPI_Allreduce(&local_ok, &ok, 1, MPI_INT, MPI_MIN, comm);
   if (ok == 0) {
      int my_rank;
      MPI_Comm_rank(comm, &my_rank);
      if (my_rank == 0) {
         fprintf(stderr, "Proc %d > In %s, %s\n", my_rank, fname, 
               message);
         fflush(stderr);
      }
      MPI_Finalize();
      exit(-1);
   }
}  /* Check_for_error */


/*-------------------------------------------------------------------
 * Function:  Read_n
 * Purpose:   Get the order of the vectors from stdin on proc 0 and
 *            broadcast to other processes.
 * In args:   my_rank:    process rank in communicator
 *            comm_sz:    number of processes in communicator
 *            comm:       communicator containing all the processes
 *                        calling Read_n
 * Out args:  n_p:        global value of n
 *            local_n_p:  local value of n = n/comm_sz
 *
 * Errors:    n should be positive and evenly divisible by comm_sz
 */
void Read_n(
      int*      n_p        /* out */,
      int*      local_n_p  /* out */,
      int*      escalar   /* out */,
      int       my_rank    /* in  */,
      int       comm_sz    /* in  */,
      MPI_Comm  comm       /* in  */) {
   int local_ok = 1;
   char *fname = "Read_n";

   if (my_rank == 0) {
      printf("What's the order of the vectors (tamaño de los vectores)?: ");
      fflush(stdout);
      scanf("%d", n_p);

      printf("What's the scalar (escalar)?: ");
      fflush(stdout);
      scanf("%d", escalar);
   }

   MPI_Bcast(n_p, 1, MPI_INT, 0, comm);
   MPI_Bcast(escalar, 1, MPI_INT, 0, comm);

   if (*n_p <= 0 || *n_p % comm_sz != 0) local_ok = 0;
   Check_for_error(local_ok, fname,
         "n should be > 0 and evenly divisible by comm_sz", comm);
         
   *local_n_p = *n_p/comm_sz;

}  /* Read_n */


/*-------------------------------------------------------------------
 * Function:  Allocate_vectors
 * Purpose:   Allocate storage for x, y, and z
 * In args:   local_n:  the size of the local vectors
 *            comm:     the communicator containing the calling processes
 * Out args:  local_x_pp, local_y_pp, local_z_pp:  pointers to memory
 *               blocks to be allocated for local vectors
 *
 * Errors:    One or more of the calls to malloc fails
 */
void Allocate_vectors(
      double**   local_x_pp  /* out */, 
      double**   local_y_pp  /* out */,
      double**   local_z_pp  /* out */, 
      double**   local_a_pp  /* out */,
      int        local_n     /* in  */,
      MPI_Comm   comm        /* in  */) {
   int local_ok = 1;
   char* fname = "Allocate_vectors";

   *local_x_pp = malloc(local_n*sizeof(double));
   *local_y_pp = malloc(local_n*sizeof(double));
   *local_z_pp = malloc(local_n*sizeof(double));
   *local_a_pp = malloc(local_n*sizeof(double));

   if (*local_x_pp == NULL || *local_y_pp == NULL || 
       *local_z_pp == NULL || *local_a_pp == NULL) local_ok = 0;
   Check_for_error(local_ok, fname, "Can't allocate local vector(s)", 
         comm);
}  /* Allocate_vectors */


/*-------------------------------------------------------------------
 * Function:   Read_vector
 * Purpose:    Read a vector from stdin on process 0 and distribute
 *             among the processes using a block distribution.
 * In args:    local_n:  size of local vectors
 *             n:        size of global vector
 *             vec_name: name of vector being read (e.g., "x")
 *             my_rank:  calling process' rank in comm
 *             comm:     communicator containing calling processes
 * Out arg:    local_a:  local vector read
 *
 * Errors:     if the malloc on process 0 for temporary storage
 *             fails the program terminates
 *
 * Note: 
 *    This function assumes a block distribution and the order
 *   of the vector evenly divisible by comm_sz.
 * 
 *  This Function was changed by Stefano Aragoni and Carol Arévalo to generate random numbers instead of reading them from stdin.
 *  MPI_Scatter was removed in order to avoid the distribution of the vector. Each process generates its own vector.
 */
void Read_vector(
      double    local_a[]   /* out */, 
      int       local_n     /* in  */, 
      int       n           /* in  */,
      char      vec_name[]  /* in  */,
      int       my_rank     /* in  */, 
      MPI_Comm  comm        /* in  */) {

   //if (my_rank == 0) {            // Antes solo el proceso con Rank=0 llenaba el array de tamaño n.
                                    // Esto se cambió para que cada proceso genere su propio vector, así 
                                    // permitiendo que se llenen varios vectores locales a la vez. 

   int i;
   for (i = 0; i < local_n; i++)
      local_a[i] = rand() % 10;   // Cada proceso almacena los números directamente en el vector local. Esto evita que solo el proceso 0
                                    // llene el vector y luego lo distribuya a los demás procesos. Es más rápido.

}  /* Read_vector */  


/*-------------------------------------------------------------------
 * Function:  Print_vector
 * Purpose:   Print a vector that has a block distribution to stdout
 * In args:   local_b:  local storage for vector to be printed
 *            local_n:  order of local vectors
 *            n:        order of global vector (local_n*comm_sz)
 *            title:    title to precede print out
 *            comm:     communicator containing processes calling
 *                      Print_vector
 *
 * Error:     if process 0 can't allocate temporary storage for
 *            the full vector, the program terminates.
 *
 * Note:
 *    Assumes order of vector is evenly divisible by the number of
 *    processes
 */
void Print_vector(
      double    local_b[]  /* in */, 
      int       local_n    /* in */, 
      int       n          /* in */, 
      char      title[]    /* in */, 
      int       my_rank    /* in */,
      MPI_Comm  comm       /* in */) {

   double* b = NULL;
   int i;
   int local_ok = 1;
   char* fname = "Print_vector";

   if (my_rank == 0) {
      b = malloc(n*sizeof(double));
      if (b == NULL) local_ok = 0;
      Check_for_error(local_ok, fname, "Can't allocate temporary vector", 
            comm);
      MPI_Gather(local_b, local_n, MPI_DOUBLE, b, local_n, MPI_DOUBLE,
            0, comm);
      printf("%s\n", title);
      for (i = 0; i < n; i++)
         printf("%f ", b[i]);
      printf("\n");
      free(b);
   } else {
      Check_for_error(local_ok, fname, "Can't allocate temporary vector", 
            comm);
      MPI_Gather(local_b, local_n, MPI_DOUBLE, b, local_n, MPI_DOUBLE, 0,
         comm);
   }
}  /* Print_vector */


/*-------------------------------------------------------------------
 * Function:  Parallel_vector_sum
 * Purpose:   Add a vector that's been distributed among the processes
 * In args:   local_x:  local storage of one of the vectors being added
 *            local_y:  local storage for the second vector being added
 *            local_n:  the number of components in local_x, local_y,
 *                      and local_z
 * Out arg:   local_z:  local storage for the sum of the two vectors
 */
void Parallel_vector_sum(
      double  local_x[]  /* in  */, 
      double  local_y[]  /* in  */, 
      double  local_z[]  /* out */, 
      int     local_n    /* in  */) {
   int local_i;

   for (local_i = 0; local_i < local_n; local_i++)
      local_z[local_i] = local_x[local_i] + local_y[local_i];
}  /* Parallel_vector_sum */



/*-------------------------------------------------------------------*/
// Implementación de las funciones adicionales


// Función que calcula el producto punto de dos vectores
double Parallel_vector_dot_product(double local_x[], double local_y[], int local_n, MPI_Comm comm) {
    double local_dot = 0.0;
    
    int local_i;

    for (local_i = 0; local_i < local_n; local_i++)
      local_dot += local_x[local_i] * local_y[local_i];

    double global_dot;
    // Se suman los productos punto de cada proceso
    MPI_Reduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Bcast(&global_dot, 1, MPI_DOUBLE, 0, comm);
    return global_dot;
}

// Función que calcula el producto escalar de un vector
void Parallel_scalar_multiply(double local_x[], double local_z[], double scalar, int local_n) {
   // Cada proceso multiplica su parte del vector por el escalar
   int local_i;

   for (local_i = 0; local_i < local_n; local_i++)
      local_z[local_i] = local_x[local_i] * scalar;
}