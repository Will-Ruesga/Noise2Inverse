## Phantom generation

**Guillem Casals and Eduard Ruiz**

This folder contains the code to generate 3D cylindrical foam phantoms based on two methods:

- **Sparse generation**. This method is very simple but less efficient. File: `sparse_generator.py`.
- **Dense generation**. This method is more complex but more efficient. File: `foam_generator.py`.

Each method is implemented in a different file, containing a class with a `run()` method to generate the phantom. Three
parameters can be regulated for each method: the number of spheres, the overlap probability and the size of the phantom,
an integer value that defines the number of pixels of the side of the phantom (3-dimensional image).

The representation of both methods for comparison is generated with `image_generation.py`.