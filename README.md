# Particles2PDEs

Source code for "Particles to PDEs Parsimoniously" by Arbabi & Kevrekidis 2020

Using first prinicples leads to microscopic evolution laws for physical, chemical and biological systems (e.g. lattice dynamics in crystals, molecular interaction in reactions or neuron response in tissue). Yet some of these systems may also admit coarse-grained evolution laws, e.g. in the form of PDEs, which can lead to huge savings in computation. We propose a frameowrk for 1) identifying the coarse-grained variable from data and 2) finding the PDE that governs that variable evolution. The example we use is a model of collective particle motion that leads to the Burgers PDE at the coarse-level description.

<img src="../master/thehood/sketch1.pdf" width="750">
<img src="../master/thehood/distances_and_moments.pdf" width="750">

## main files:

**gap_tooth** illustrates the gap-tooth scheme for parsimonious simulations of particle dynamics (Fig. 1 in the paper).

**learn_PDE_density**  builds and trains neural nets that learn the right-hand-side of PDEs for the denisty (*a priori* known coarse-grained variable).

**Variable Identification** uses unnormalized/unbalanced optimal transport distance + diffusion maps to learn the coarse-grained variable from particle data.

**learn_PDE_phi** builds and trains neural nets that learn the right-hand-side of PDEs for the denisty (coarse-grained variable discovered from data).

## files under 'thehood':

**unbalanced_transport:** implementations of Chizat *et al.* 2018 and Gangbo *et al.* 2019 (analytical) formulations of unbalanced optimal transport.

**BurgersGapTooth:** the gap-tooth implementation of the particle model leading to Burgers PDE.

**model_library:** collection of neural net models for learning PDEs.

## dependencies

[TensorFlow >=2.0](https://www.tensorflow.org/install)

