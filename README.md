# COL380_A2
CUDA-parallelized CNN for recognizing MNIST digits

## Running the code:

- Subtask 1:
  - Compile the code as follows: ```make subtask1```
  - Run the program with necessary arguments:
  - task of choice 1=convolution, 2=non-linear-activations, 3=subsampling, 4=converting a vector
  - For choice 1:   
  - ```./subtask1 1 <input_dim> <kern_dim> <pad_val> <input_matrix_entries> <kernel entries></kernel>```
  - For choice 2: (0 - relu, 1 - tanh)
  - ```./subtask1 2 <activation_function> <N_dim> <M_dim> <input_matrix_entries>```
  - For choice 3: (0 - maxpool, 1 - avgpool)
  - ```./subtask1 3 <poolfunc> <pool_size> <input_dim> <input_matrix_entries>```
  - For choice 4: (0 - sigmoid, 1 - avgpool)
  - ```./subtask1 4 <prob_func> <input_vector_entries>```

- Subtask2:
  - Compile the code as follows: ```make subtask2```
  - Run program with nessary arguments as in subtask1

- Subtask3:
  - Make sure to have /img directory with required images
  - Compile the code as follows: ```make subtask3```
  - Run program as ```./subtask3```

- Subtask4:
  - Make sure to have /img directory with required images
  - Compile the code as follows: ```make subtask4```
  - Run program as ```./subtask4 [1 - with streams, 0 - without streams]```

