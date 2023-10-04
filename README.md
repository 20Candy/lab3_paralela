# Laboratorio 3 Paralela
Implementación de operaciones vectoriales utilizando el paradigma de programación de paso de mensajes (MPI).


## Instalacion 🔧

1. Requisitos previos: Asegúrate de tener instalado el paquete MPI para poder compilar y ejecutar programas MPI. Si aún no lo tienes instalado, puedes hacerlo con los siguientes comandos: 

```shell
    pysudo apt update
    sudo apt install -y openmpi-bin openmpi-common libopenmpi-dev
```

2. Clona este repositorio:

```shell
    https://github.com/20Candy/lab3_paralela
```


3. Accede al directorio del proyecto 
```shell
    cd [nombre_del_directorio]
```

## Correr el programa 🚀

1. Programa 1

a. Compilación

```shell
  gcc -g -Wall -o vector_add vector_add.c
```

b. Ejecución
```shell
  ./vector_add
```


2. Programa 2

a. Compilación

```shell
  mpicc -g -Wall -o mpi_vector_add mpi_vector_add.c
```

b. Ejecución
```shell
  mpiexec -n [NUMERO_DE_PROCESOS] ./mpi_vector_add
```

* Nota: Reemplaza [NUMERO_DE_PROCESOS] con el número de procesos que desees utilizar para la ejecución.

3. Programa 3

a. Compilación

```shell
  mpicc -g -Wall -o mpi_vector_add_2 mpi_vector_add_2.c
```

b. Ejecución
```shell
  mpiexec -n [NUMERO_DE_PROCESOS] ./mpi_vector_add_2
```

* Nota: Reemplaza [NUMERO_DE_PROCESOS] con el número de procesos que desees utilizar para la ejecución.


## Construido con 🛠️
- C
- MPI (Message Passing Interface)

## Características 📋

Este programa realiza las siguientes operaciones sobre vectores:

- Producto punto de dos vectores.
- Multiplicación de un vector por un escalar.

## Autores ✒️

* **Carol Arevalo** - *desarrollo* - [#20Candy](https://github.com/20Candy)

* **Stefano Aragoni** - *desarrollo* - [#stefanoaragoni](https://github.com/stefanoaragoni)


