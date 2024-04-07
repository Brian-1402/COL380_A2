# COL380_A2
CUDA-parallelized CNN for recognizing MNIST digits

## Todo

### General

- Read piazza and fully understand submission and directory format.
- See how the weights are stored and how to import it.
- See how the assignment is tested.

### Subtask 1

- may have to rewrite function definitions to use arrays instead of vectors.
- Look into fast convolution or whatever.

### Subtask 2

- Maybe look into shared memory optimization for convolutions and pooling.

### Other possible optimizations
- Summing, pooling, etc and other serial operations can be parallelized using [reduction kernel](https://shreeraman-ak.medium.com/parallel-reduction-with-cuda-d0ae10c1ae2c).

