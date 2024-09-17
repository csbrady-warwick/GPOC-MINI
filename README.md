# GPOC++-MINI

GPOC++-MINI is a truly minimal miniapp implementing the EPOCH (https://github.com/Warwick-Plasma/epoch) PIC algorithm in a simple CUDA package. The current version is a trivial implementation operating entirely out of shared memory, and solves a simple problem of a periodic box with a laser driven from both ends interacting with a slab of electrons in the middle of the domain. It uses particle sorting to achieve decent performance on workstation cards (such as the A6000), but will generally achieve at best 10% of peak compute on datacentre cards such as the A100 or H100

This code is NOT intended to be a working PIC code, but to demonstrate the problems that will be encountered by a real PIC code when running on a GPU and as a basis for developing more sophisticated models for GPU PIC development using the core EPOCH algorithm.

As this code becomes more sophisticated a Makefile will be used but at the moment, simply build it using

nvcc -arch=sm\_80  -I include/ -O3 src/em.cu -o em

sm\_80 is the minimum needed for this code to build. There are options that can be set as precompiler flags. Specify the following on the compile line to turn on and off features

-DNOSORT  - Run the code without any sorting of the particles. Performance will be much lower, showing the importance of memory locality

-DWITHIO - Causes the code to write a PPM image of the density of the electron plasma periodically. Can be used to check correctness of answers or for "attract" images
