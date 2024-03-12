# 0: Introduction to this document
This is quick write-up meant to explain the methodology and methods in the data driven indexing approach. This document will serve as the basis for an eventual publication. The current status is word vomit.

I am very interested in feed-back on the methods. I feel that there is large room for improvement in the Miller index assignment and optimization aspects of this algorithm. I have devoted an enormous amount of time to the unit cell regression aspect of this algorithm and I don't believe there is much room for improvement there. There is no refinement what-so-ever. No hyperparameter training, equal amounts of under and over fitting.

Please note, text in italics are commentary. For example, my hunches, opinions, acknowledgements of error, or features and approaches I would like to take, and features that are implemented but not used. Normal case text describes currently implemented methods

# 1: Introduction
Powder diffraction indexing is the process of using a 1D projection of diffraction to infer the symmetries that exist within a material. This is a problem as old as crystallography and has been the subject of many analytical efforts during the development of crystallography. Current algorithms in use take a generate-and-test approach to the solution. Current trends in machine learning and artificial intelligence have produced numerous new tools for data driven analytics. This project merges recent advancements in data driven analytics with knowledge from existing indexing algorithms to produce a novel, modern approach to powder diffraction indexing.

## 1.1 Powder diffraction indexing
Diffraction is produced in a 3-dimensional spherical coordinates - reciprocal space. In the powder diffraction method, the 3d diffraction pattern is recorded in the radial dimension, the distance of the measured diffraction from the center of the sphere. The recorded pattern is a series of peaks with the location of the peaks representing the spacing of Miller planes through the crystal's unit cell. In the indexing problem, the objective is to determine the atomic-level symmetries of the crystal that produced this pattern. When the diffraction is measured in coordinates of $q = \frac{2\sin(\theta)}{\lambda}$, where $\theta$ is the diffraction angle and $\lambda$ is the wavelength, the crystal's unit cell can be related to the observed peak locations with through the general equation:

$q_{hkl}^2 = \frac{1}{V}\left(S_{hh}h^2 + S_{kk}k^2 + S_{ll}l^2 + S_{hk}hk + S_{hl}hl + S_{kl}kl\right)$,

$S_{hh} = b^2 c^2 \sin^2(\alpha)$,

$S_{kk} = a^2 c^2 \sin^2(\beta)$,

$S_{ll} = a^2 b^2 \sin^2(\gamma)$,

$S_{hk} = a b c^2 \left[\cos(\alpha)\cos(\beta) - \cos(\gamma)\right]$,

$S_{hl} = a^2 b c \left[\cos(\beta)\cos(\gamma) - \cos(\alpha)\right]$,

$S_{kl} = a b^2 c \left[\cos(\alpha)\cos(\gamma) - \cos(\beta)\right]$,

$V (\mathrm{Volume}) = a b c \sqrt{1 - \cos^2(\alpha) - \cos^2(\beta) - \cos^2(\gamma) + 2\cos(\alpha)\cos(\beta)\cos(\gamma)}$.

This general equation assumes no simplifying constraints among the crystal lattice lengths and angles. This general equation can be simplified given constraints amongs the lattice parameters defining the seven lattice systems:

$\mathrm{cubic}: a=b=c, \ \alpha=\beta=\gamma=90^o$

$\ \ \ \ q_{hkl}^2 = \frac{h^2 + k^2 + l^2}{a^2}$

$\mathrm{tetragonal}: a=b\neq c, \ \alpha=\beta=\gamma=90^o$

$\ \ \ \ q_{hkl}^2 = \frac{h^2 + k^2}{a^2} + \frac{l^2}{c^2}$

$\mathrm{hexagonal}: a=b\neq c, \ \alpha=\beta=90^o\ \gamma=120^o$

$\ \ \ \ q_{hkl}^2 = \frac{4}{3}\frac{h^2 + hk + k^2}{a^2} + \frac{l^2}{c^2}$

$\mathrm{rhombohedral}: a=b=c, \ \alpha=\beta=\gamma\neq90^o$

$\ \ \ \ q_{hkl}^2 = \frac{(h^2 + k^2 + l^2)\sin^2(\alpha) + 2(hk + kl + hl)\cos^2(\alpha) - \cos(\alpha)}{a^2(1-3\cos^2(\alpha) + 2\cos^3(\alpha)}$

$\mathrm{orthorhombic}: a\neq b\neq c, \ \alpha=\beta=\gamma=90^o$

$\ \ \ \ q_{hkl}^2 = \frac{h^2}{a^2} + \frac{k^2}{b^2} + \frac{l^2}{c^2}$

$\mathrm{monoclinic}: a\neq b\neq c, \ \alpha=\gamma=90^o, \beta\neq 90$

$\ \ \ \ q_{hkl}^2 = \frac{h^2}{a^2\sin^2(\beta)} + \frac{k^2}{b^2} + \frac{l^2}{c^2\sin^2(\beta)} - \frac{2hl\cos(\beta)}{ac\sin^2(\beta)}$.

The general case is known as the triclinic. 

The powder diffraction indexing problem usually starts with a set of 20 observed peak positions $\{q_{obs}\}_{hkl}$. The lattice system equations give the relationship between each peak in the set to the same six lattice parameters, however, each of the 20 equations is parameterized by a unique set of Miller indices $\{hkl\}$. If $n$ peaks are choosen, there will be between $3n + 1$ and $3n + 6$ unknown parameters, there will always be three times more unknown parameters than equations. This is an underdetermined inverse problem - more parameters being predicted than constraints that can be imposed.

For cubic lattice systems, the single free lattice parameter, $a$ can be found rather simply by pencil and paper, each peak occurs at an integer value divided by $a^2$. For tetragonal, hexagonal, and rhombohedral with two free lattice parameters, solutions can be reliably found without computation aids through graphical methods (CITE). Difficulty begins with orthorhombic cases of three free lattice parameter, however computational methods have been found to reliably produce results (CITE). Monoclinic and triclinic lattice systems are known to difficult for computational algorithms.

## 1.2 Overview of existing powder diffraction indexing algorithms

The method presented here draws inspiration heavily from the Iterative SVD (SVD-Index) indexing algorithm presented by Coehlo (2003). This algorithm takes a generate-and-test approach to the problem, it starts with random guess's of the lattice parameters, then iteratively assigns Miller indices to each peak and optimizes the lattice parameters to explain the observed diffraction spacings.

This is a complex algorithm and an abridged overview is given here to support the development of the data-driven indexing approach. SVD-Index consists of three primary components: 1) initial unit cell generation; 2) Miller index assignment; 3) Unit cell optimization. 

SVD-Index generates initial unit cell candidates by randomly generating numbers for the unit cell parameters, then rescaling the parameters to match a given unit cell volume. The algorithm incrementally increases the unit cell volume that is used for rescaling until a solution is found. 

The Miller index assignment step starts by establishing a reference set of Miller indices. This is done by calculating all possible Miller indices up to a certain resolution cutoff given a unit cell volume. For each reference Miller index and candidate unit cell, the forward models (Eq. X - Y), are used to calculate reference peak spacings, $\{q_{ref}\}_{hkl}$. Observed peaks are assigned miller indices associated with the closest peak position in the set of reference peak spacings.

Peak spacings can be calculated for each for each peak given a candidate unit cell and Miller index assigment, $\{q_{cal}\}_{hkl}$. Unit cell parameters are optimized to match the agreement between the observed, $\{q_{obs}\}_{hkl}$, and calculated, $\{q_{cal}\}_{hkl}$. peak spacings. This is performed using weighted least squares solved by singular value decomposition (SVD). The forward models (Eq. X - Y) can be reformulated as a linear function of Miller indices and transformed unit cell parameters, $\vec{q}_{cal} = H\vec{x}$. The least squares solution to $\vec{x}$ to $w_{hkl}\vec{q}_{obs} - w_{hkl}H\vec{x}$, where $w$ is a set of weights for each peak, is then found using SVD (CITE). The weight function is 

$w_{hkl} = d_{obs,hkl}^m|2\theta_{obs,hkl} - 2\theta_{cal,hkl}|I_{obs}$.

Here, $d_{obs,hkl}$ and $2\theta_{obs,hkl}$ are the observed peak spacings in resolution (d-spacing: $d = 1/q$) and angular units. $2\theta_{cal,hkl}$ is the calculated peak spacing in angular units. $I_{obs}$ is the peaks integrated intensity. The ideal exponent, $m$, was noted to be 4, however using a random number generated between 0 and 4 was found to be the superior choice. 

This algorithm works by assuming a lattice system 

## 1.3 Data-driven indexing approach
The data-driven indexing approach seeks to improve the three primary components of SVD-Index using machine learning, maximum-likelihood optimization, and Monte-Carlo inspired randomization.

Machine learning methods are used to create a random unit cell generators with the goal of reducing the initial search space. These are either neural networks that predict a distribution of the unit cell parameters that can be sampled from, or random forest models made of individual decision trees that can be sampled from. For each lattice system, an ensemble of machine learning models are trained on data separated by unresolvable unit cell ambiguities or expected differences in peak spacing "distributions". For a given lattice system, there are ambiguities in the unit cell that is being predicted. For example, for an orthorhombic lattice, which unit cell axis is the shortest and which is the longest? This ambiguity can be easily resolved by reindexing each entry in the training data so axis $a$ is the shortest and $c$ is the longest. The same reindexig can be performed for monoclinic, but this leads to an unresolvable ambiguity for which angle is not set at $90^o$. Also for a given lattice system, there will be different "distributions" of observed peak spacings due to different patterns of systematic absences.

Miller index assignments are created by first establishing a set of reference Miller indices, these are simply the first few hundred Miller indices that provide unique peak spacings for a given lattice system. An array of pairwise differences between the observed and calculated peak positions, $q^2_{obs, hkl} - q^2_{ref, hkl}$ is used as inputs to a neural network classifier that produces a probability distribution over the possible Miller indices. Choosing the most probable Miller index is more accurate that choosing the closest reference Miller index, and the probabilty of assignment is useful for the optimization component.

Optimization of the candidate unit cells given assigned Miller indices is performed using maximum-likelihood optimization, seeking the most probable unit cell parameters and Miller index assignments given the observed diffraction, $\rho(H,\vec{c}|\vec{q}_{obs})$. Then Miller indices are reassigned and the optimization is repeated, in an iterative manner. Randomization is applied to the optimization by randomly subsampling peaks in each optimization trial. New unit cell parameters are accepted in the same manner as the Metropolis algorithm. The ratio of the next to current likelihoods are calculated and accepted if a randomly generated number between 0 and 1 is less than the ratio.

# 2: Methods

## 2.1 Data

### 2.1 Preparation

### 2.2 Grouping

### 2.3 Augmentation

### 2.2 Regression models

### 2.2.1 Preprocessing

### 2.2.2 Neural networks

### 2.2.3 Random forest

### 2.2.4 Evaluations

### 2.3 Assignment models

### 2.3.1 Preprocessing

### 2.3.2 Neural network

### 2.3.3 Evaluations

### 2.4 Optimization

### 2.4.1 Target function

$\rho(H,\vec{c}|\vec{q}_{obs}) = \frac{\rho(\vec{q}_{obs}|H, \vec{c})\rho(H, \vec{c})}{\rho(\vec{q}_{obs})}$.

The denominator can be dropped because it will be constant for all unit cell paramters,

$\rho(H,\vec{c}|\vec{q}_{obs}) \propto \rho(\vec{q}_{obs}|H, \vec{c})\rho(H, \vec{c})$.

Miller index assignments, $H$ are conditional on the unit cell parameters,

$\rho(H,\vec{c}|\vec{q}_{obs}) \propto \rho(\vec{q}_{obs}|H, \vec{c})\rho(H| \vec{c})\rho(\vec{c})$.

This can be written as a product of probabilites over each observed peak spacing $q_{obs,k}$, assuming there are $n_k$ peaks and an independence between peaks.

$\rho(H,\vec{c}|\vec{q}_{obs}) \propto \Pi_{k=1}^{n_k}\left[\rho(q_{obs,k}|H, \vec{c})\rho(H|\vec{c})\right]\rho(\vec{c})$.

This equation can be further expanded as a summation over all $n_h$ reference Miller indices

$\rho(H,\vec{c}|\vec{q}_{obs}) \propto \Pi_{k=1}^{n_k}\left[\Sigma_{h=0}^{n_h}\rho(q_{obs,k}|H_h, \vec{c})\rho(H_h|\vec{c})\right]\rho(\vec{c})$.

### 2.4.2 Random unit cell generation

### 2.4.3 Subsampling

### 2.4.4 Optimization

### 2.4.5 Acceptance

### 2.4.6 Found explainers

# 3: Results

# 4: References

Coelho, A. A., (2003) J. Appl. Cryst. 36, 86-95.
        