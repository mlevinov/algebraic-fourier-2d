# <p style="text-align:center;"> Algebraic Reconstruction of Piecewise-Smooth Functions of Two Variables from Fourier Data </p>
#### <p style="text-align:center;"> Thesis submitted in partial fulfilment of the requirements for the M.Sc. degree of Tel Aviv University by Levinov Michael under the supervision of Dr. Batnekov </p>

### Description:
This is the implementation of the *2D algorithm* give on pages 42 which relies on the [mpmath](https://mpmath.org/) python library for arbitrary precision calculations.  
  
* The user will be able to recreate the results shown in *Numerical Experiments* chapter (p.42).
* The *single_jump_param_recovery.py* file contains the methods for the reconstruction of a 2D function that meets the required assumptions given in *Definition 3*.
* The *mpmath_tools.py* contains methods that were lacking in the [mpmath](https://mpmath.org/) library, such as converting an mpmath.matrix to a numpy array.
* The *test_function.py* file contains a class that represents three test functions for whom their Fourier coefficients were calculated analytically and are used for the described experiment.
  **The functions are:**  
  * $F_1(x,y)$ has a discontinuity curve $\xi(x) = x$ and jump magnitudes of $\frac{d}{dx}F_1 $ at each $x\in [-\pi, \pi)$ are given by $\frac{x}{2\pi(d+1)}$ as $d = 0,\ldots, 5$.
  * $F_2(x,y)$ has a discontinuity curve $\xi(x) = x$ and jump magnitudes of $\frac{d}{dx}F_2 $ at each $x\in [-\pi, \pi)$ are equal to 1 (constant) as $d = 0,\ldots, 11$.
  * $F_3(x,y)$ has a discontinuity curve $\xi(x) = \frac{x}{2}$ and jump magnitudes of $\frac{d}{dx}F_3 $ at each $x\in [-\pi, \pi)$ are equal to 1 (constant) as $d = 0,\ldots, 11$.
* The *plot_tools.py* file contains the implementation of the algorithm on the test functions and provides the data for the *plots.py* file and can be used as an example for an implementation.

