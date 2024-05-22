## Phantom generation

**Guillem Casals and Eduard Ruiz**

This folder contains the code to generate 3D cylindrical foam phantoms based on two methods:

- **Sparse generation**. This method is very simple but less efficient. File: `sparse_generator.py`.
- **Dense generation**. This method is more complex but more efficient. File: `foam_generator.py`.

Each method is implemented in a different file, containing a class with a `create_phantom()` method to generate the phantom. Three
parameters can be regulated for each method: the number of spheres, the overlap probability and the size of the phantom,
an integer value that defines the number of pixels of the side of the phantom (3-dimensional image).

The representation of both methods for comparison is generated with `img_phantom_comparison.py`.

## Sinogram generation

**Guillem Casals and Eduard Ruiz**

This project contains the `foam_generator.py` file with the `FoamGenerator` class to generate the phantom. The `sinogram_generator.py` file contains the code to generate a sinogram, add noise and reconstruct. Different representations are generated to visualize the sinograms and reconstructions with certain degrees of noise.

To run the code, run through the terminal:

```
python3 sinogram_generator.py
```

It is important to have an environment with the Astra toolbox installed in order to run the code.