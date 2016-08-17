# Computational Geometry

Projects for the Computational Geometry and Machine Learning course developed in Python.


## Modules

### fisher.py

The class Fisher offers methods to train and classify data using the Fisher's Linear Discriminant. Based on Chapter 4.1 from C.M. Bishop, Pattern Recognition and Machine Learning, Springer, 2007.

### reduce_dim.py

This module implements two mecanisms for dimensionality reduction. The first one is LDA, which performs the Linear Discrimant Analysis (based on Section 4.1.6 from C.M. Bishop, Pattern Recognition and Machine Learning, Springer, 2007), and the second is PCA, which implements the Principal Component Analysis (based on Section 12.1 from the same book).
        
### bezier.py

This module implements several algorithms to evaluate Bezier curves. For further references, please see Chapters 2 and 3 from Bézier and B-Spline Techniques by Prautzsch, H., Boehm, W., Paluszny, M.

### Interactive.ipynb

This notebook makes it possible to play with two Bézier curves. You can add or move control points of one curve using left click, and right click for the second curve. It also computes their intersections.

### Surfaces.ipynb

This notebook renders a Bézier surface given its control points. As an example, we render the typical Utah teapot.

<img src=teapot.png width=618 height=440 alt='Utah teapot' />

### Convex Hull.ipynb

This notebook implements the Graham's algorithm to compute the convex hull of a set of points in the plane.

### Convex Hull 3D.ipynb

This notebook implements a beneath-beyond algorithm to compute the convex hull of a set of points in the space. It can also compute Delaunay triangulations in 2D using a trick.
	
	
## Authors

This project is being developed by Ana María Martínez Gómez and Víctor Adolfo Gallego Alcalá. 



## Licence

Code published under MIT License (see [LICENSE](LICENSE)).
