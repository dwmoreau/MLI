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
Crystallographic data for training the machine learning models are sourced from the Cambridge Structural Database (CSD) and the Crystallographic Open Database (COD). The databases do not contain experimental data, they contain structural information about materials that were determined experimentally. Powder diffraction patterns and d-spacings can be calculated from these databases.

True experimental data, as in experimentally measured powder diffraction patterns are sourced from the RRUFF database (Lafuente, et. al., 2015).

### 2.1.1 Preparation
The first step of the dataset generation process is an initial quality control of entries in the CSD and COD. The following criteria must be met by an entry for its inclusion in either training or validating the machine learning models.

1. Chemical formula must be listed.
2. Spacegroup number must be listed and between 1 and 230, inclusively.
3. The Hermann–Mauguin symbol must be obtainable from the listed spacegroup symbol.
4. Crystal system must be listed and compatible with the given conventional unit cell parameters.
5. Unit cell must be able to be converted from the conventional to reduced setting.
6. Volume calculated from the given unit cell parameters must match the given volume.
7. Volume should be consistent with the number of atoms in the unit cell deduced from the chemical formula. The given volume in $\AA^3$ must be within 18 times the number of atoms in the unit cell.

The next step is a duplicate removal step performed on the CSD and COD independently. A duplicate is defined as the following:

1. Belongs to the same crystal family. These are the same as lattice systems presented in Eq. X-Y, except with hexagonal and rhombohedral grouped together.
2. Unit cell has the same chemical composition deduced from the chemical formula.
3. Unit cell is within 5%.

In the case of duplicate entries, the entry with the lowest reported r-factor is choosen.

$\textit{Also, I just noticed a bug in the code and the refinement r-factor is not being retained, currently a random duplicate is choosen.}$

Databases are merged by first taking all the unique entries from the CSD, then only adding entries from the COD if they are not duplicated with an existing CSD entry.

$\textit{I would like to be a bit more strict with the duplicate removal process, so condition 2 needs to change to the chemical composition differs by at most one}$
$\textit{atom. There are many studies that test the effects of changing one atom in a material. These entries are essentially the same unit cell but the contents}$
$\textit{differ by one atom. These studies will contribute to data leakage, essentially identical entries in the training and validation sets. I get the best}$
$\textit{performance when I use wide shallow neural networks, this was also observed by Vanessa. My understanding of wide shallow neural networks are they tend to}$
$\textit{work more by memorization as opposed to generalization. My concern is that models are memorizing identical patterns as opposed to generalization.}$


Datasets are generated from the unique entries in both the CSD and COD using the CCDI api. The following information is extracted from each entry:

* Diffraction pattern
* Database source
* Identifier (If from CSD)
* Local cif file name (If from COD)
* Spacegroup number
* Bravais Lattice
* Lattice system
* Hermann–Mauguin spacegroup symbol$^{i}$
* Unit cell parameters$^{i,r}$
* Unit cell volume$^{r}$
* Peak lists$^{i}$

The diffraction pattern is calculated between $2\theta = 0^o -> 60^o$ in steps of $0.02^o$ with peak breadths of 0.1$^o$ full width at half max. It is calculated assuming the Cu K-$\alpha$ wavelength of 1.54 $\mathrm{\AA}$. It is used to determine the peak list and is retained, although not used after dataset generation.

Two peak lists are included, these are 1) the first 60 non-systematically absent peaks and 2) the first 60 peaks that could be realistically observed in a diffraction pattern. For a diffraction peak to be included in peak list 2, each of the following conditions must be met:

1. Not systematically absent.
2. The diffraction pattern is normalized such that $\sqrt{\Sigma I^2} = 1$. The intensity at the peak position of the normalized diffraction pattern must be larger that 0.001.
3. The second derivative of the diffraction pattern at the peak position must be negative
4. The peak must not be within 0.1$^o$ / 1.5 of another peak. If so, the peak with the largest intensity is retained.

Three different representations of unit cell are included, the convention unit cell, which is taken directly from the the database. Next is a reindexed unit cell. For orthorhombic, monoclinic, and triclinic, the conventional unit cell is reindexed so the unit cell lengths, $a,\ b,\ c$ are in increasing lengths. This results in changes to the Hermann–Mauguin spacegroup symbol, unit cell parameters, and Miller indices in the peak lists. Last is the Niggli reduced unit cell, when results in chages to the unit cell parameters and volume.

### 2.1.2 Grouping
Each unit cell prediction model is based on a different set of grouped entries. The groupings are performed with this heirarchy:

1. Lattice system, then
2. Bravais lattice, then
3. Patterns in reflection conditions.

Grouping heirarchies 1 and 2 are well defined, heirarchy 3 is subjective. To be clear what is meant by grouping by patterns in reflection conditions, consider the primitive orthorhombic space groups P 2 2 2 and P 21 21 21. P 2 2 2 has no systematic absences. The P 21 21 21 spacegroup has screw axis symmetry along the $a$, $b$, and $c$ axes, leading to the reflection conditions that Miller index $h$ must be even when $k = l = 0$; the $h00$ reflection must have an even $h$. The same condition is imposed on $k$ and $l$ for $0k0$ and $00l$ respectively. In the case that a mathematical model that determines the orthorhombic unit cell parameters, and does so by placing all of its emphasis on the low angle peaks which are more likely have Miller indices $100$, $010$, and $001$, It would first need to identify the difference between the P 2 2 2 and P 21 21 21 entries before predicting the unit cell parameters. If it cannot distinguish these cases, it should have great difficulty in the predicting the unit cell parameters.

It should be noted that heirarchy 3 splits up entries with the same space group. Consider spacegroup 18 which has two screw axes and one axis without a symmetry element. This one spacegroup contain spacegroup symbols P 21 21 2, P 21 2 21, and P 2 21 21. The P 21 21 2 spacegroup will have two reflection conditions $h00; h=2n$ and $0k0; k=2n$, and no reflection condition for $l$. The other two space symbols represent different permutations of the axes, and reflection condition. Again, orthorhombic crystals are reindexeds so $a<b<c$, so all three settings will be contained in the training/testing datasets. Spacegroup 18 is split into three different groups, where they will be grouped with entries from other spacegroups with similar reflection conditions. On the otherhand, consider spacegroup 17, with one screw axis and two axes without a symmetry element. This spacegroup contains symbols P 21 2 2, P 2 21 2 and P 2 2 21. Ideally, they would be split into different groups like spacegroup 18, however, spacegroup 17, and other spacegroups with similar patterns in reflection conditions, have very few entries, so they are grouped together into a single group.

Specifications for the different groups can be found at "data/GroupSpec_$\textit{lattice_system}$.xlsx"

$\textit{I have a hunch that this grouping will help with generalization. Absent peaks, for whatever reason, should look like systematic}$
$\textit{absent patterns in a different group, so there should always be a random unit cell generator that matches the observed}$
$\textit{patterns in missing peaks, regardless if they are from systematic absences or other reasons}$

### 2.1.3 Training / Validation split

A 80 / 20% split is made for the training and validation sets. This split occurs at the level of the grouped entries. So the testing / validation split is performed on each individual group at grouping heirarchy 3.

$\textit{Splitting up the training and validation sets at the spacegroup symbol level seems to make more sense}$

### 2.1.4 Parameter scaling

Neural networks perform best with inputs and outputs that are on the scale of one. Training occurs considering each lattice system at a time. For a given lattice system, the observed peak spacing as $\vec{q}_{obs}^2$, are scaled using a standard scaler. Meaning the mean and standard deviation of all $\vec{q}_{obs}^2$, so for all entries in the training set and peak positions. Then the mean is subtracted from each peak spacing, then the difference is divided by the standard deviation.

Unit cell length parameters are also scaled using a standard scaler. For a given lattice system, the mean and standard deviation of all the unit cell lengths of all the entries in the training, are calculated and used for scaling.

Unit cell angle parameters are converted to radians and subtracted by $\pi/2$. This puts right angles ($90^o$) at zero. Then the standard deviation of all angles, which were not originally $90^o$, is calculated in radians, and the zero centered unit cell angles are divided by the scale.

### 2.1.5 Augmentation

The grouping approach requires a broad diversity of data, as opposed to just simply more data. A broader range of data allows for more groups. To increase the breadth of data, augmentation is performed, preferentially augmenting entries associated with Hermann–Mauguin spacegroup symbols with few entries. Augmentation happens after splitting the dataset into training and validation sets. If the original entry is in training set, all entries augmented from this entry will remain in the training set, and visa-versa for the validation set. A single entry is augmented at most 10 times. 

There are two augmentations that need to be performed for each entry. First, the unit cell parameters are randomly perturbed. Second, the observed peaks are resampled from the non-systamically absent peaks. There are three implemented methods to randomly perturb the unit cell parameters, all of these perturbations occur in the scaled unit cell parameter space:

1) Random perturbations. A unit cell parameter for an entry is perturbed by randomly sampled from a Normal distribution with the original entry's unit cell parameter as a mean and 0.2 as the standard deviation.

2) Covariance perturbations. The covariance of the scaled unit cell parameters are calculated over all the training entries. Scaled unit cell parameters for an entry are perturbed by sampling from a multivariate normal distribution centered at the unperturbed scaled unit cell parameters and with a covariance matrix equal to $0.2^2$ times the covariance calculated over the training set.

3) SVD perturbations. A SVD is performed on the scaled unit cell parameters from the training set. The training set's unit cell parameters are then transformed into the new orthogonal basis from the SVD and the standard deviation of each component is calculated. An entry's unit cells are perturbed by transforming the scaled unit cell parameters using the SVD, then sampling new components from a Normal distribution centered at the entries transformed component values and with standard deviation equal to 0.2 times the standard deviation of that component calculated over the training set.

The random perturbation method is used only for the cubic lattice system because a covariance or SVD are not possible with one parameter. The covariance and SVD methods were implemented so any correlation between unit cell parameters observed in the training data will be retained in the augmented entries. The covariance method is implemented in the code, but not used. All lattice systems other than cubic use the SVD perturbation method.

The peak list is augmented by resampling peaks from the list of non-systematically absent peaks. This resampling method was developed so the distribution of peaks in the augmented entries matched the distribution of the unaugmented entries.

There are two peak lists available at this time; Peak list 1, the first 60 non-systematically absent peaks. Peak list 2, the first 60 peaks that could be realistically observed in a diffraction pattern. Again, only using the training data to setup this augmention. 

The position of the first observed peak in the list of all non-systematically absent peaks is recorded. An empircal distribution is made from a histogram of this position list. For the augmented entries, the empirical distribution is used to randomly select a first peak from the list of all non-systematically absent peaks. 

After the first peak is picked, the list of non-systematically absent peaks is stepped through and the next peak is picked based on its distance from the current selected peaks. Selecting the next peak based only on its separation from the current peak is choosen to make observability only based on whether or not the peak will overlap another peak. Intensity of the peak is not a consideration. This approach is set up by calculating the separation, $\Delta q^2$ in units of $q^2$, between each neighboring peak in Peak list 2, the realistic peak list. The fraction of neighboring peaks in Peak list 2 that are also neighboring in peak list 1 are calculated in binned units of separation. This binning is in 100 bins log between $10^{-4}$ to $10^{-1.75}$, with log-spacing. A function of the form, $\rho(\Delta q^2) = \left[1 - \exp(-r_0 \Delta q^2)\right]^{r_1}$ is fit to the binned fraction and is used as the acceptance probability of the next peak based on its separation from the current peak. This form was choosen because when $\Delta q^2 = 0$ the probability of acceptance is zero and as $\Delta q^2 => \inf$, the probability of acceptance approaches one. The probability increases monotonicaly between these two extremes.

### 2.2 Regression models

Regression models are developed with the goal of using their outputs to randomly generate unit cell candidates based on a set of peaks. For all lattice systems except cubic, 20 peaks from peak list 2 are used. For cubic lattice systems, 10 peaks are used because there tends to be fewer observable peaks in the diffraction patterns and 10 peaks is more than sufficient for one free parameter. 

### 2.2.1 Preprocessing

### 2.2.2 Neural networks

Regression neural networks are setup as variance networks. They output a mean and variance and are trained using a negative log-likelihood target function, which assumes a Normal likelihood. The approach presented here are based heavily on recent research on Variance networks (Detlefsen, et. al., 2019; Seitzer, et. al., 2022; Stirn & Knowles, 2021; Stirn et. al., 2022).

1) A distribution is predicted for variance as opposed to a point estimate. The Inverse-Gamma distribution is the posterior distribution for the unknown variance of a Normal distribution and is parameterized by $\alpha$ and $\beta$. The neural network outputs three values for each unit cell parameter, mean$=\mu$, $\alpha$ and $\beta$.
2) Each model contains three parallel neural networks outputing $\mu$, $\alpha$ and $\beta$ seperately. These networks start independently from the input peak spacings are have no connections with each other.
3) The training occurs in a cyclic manner. First, the weights of the $\mu$ network are optimized while the weights of the $\alpha$ and $\beta$ networks are held fixed. Then the weights of the $\mu$ network are fixed while the weights of the $\alpha$ and $\beta$ networks are optimized. Generally, each cycle is run for 10 epochs and five cycles are performed.
4) A $\beta$-NLL target function is used when training the $\mu$ network and a NLL target function is used when training the $\alpha$ and $\beta$ networks. The NLL target function is defined as:

$L_{NLL} = \Sigma_{i=0}^{n_{uc}}\ \frac{1}{2}\ln(2\pi) - \alpha_i\ln(\beta_i) - \ln(\Gamma(\alpha_i + 1/2)) + \ln(\Gamma(\alpha_i)) + (\alpha_i + 1/2)\log(\beta + \frac{1}{2}(uc_i - \hat{uc}_i)^2)$.

Here, the summation occurs over the unit cell parameters, $uc_i$ are the true unit cell parameters, and $\hat{uc}_i$ is the estimated mean. The $\beta$-NLL target function multiplies the NLL target functio by the variance estimate raised to the power $\beta$. This multiplicative term is excluded from the gradients calculations, as denoted by the stop-gradient operator $\lfloor\cdot\rfloor$,

$L_{\beta-NLL} = \Sigma_{i=0}^{n_{uc}}\ \left\lfloor \left(\frac{\beta}{\alpha - 1} \right)^{\beta*} \right\rfloor \left[\frac{1}{2}\ln(2\pi) - \alpha_i\ln(\beta_i) - \ln(\Gamma(\alpha_i + 1/2)) + \ln(\Gamma(\alpha_i)) + (\alpha_i + 1/2)\log(\beta + \frac{1}{2}(uc_i - \hat{uc}_i)^2)\right]$.

In the Inverse-Gamma distribution estimate of variance, the variance is estimated as $\frac{\beta}{\alpha - 1}$ where $\alpha > 1$. The $\beta$-NLL target function raises this value to the power of $\beta* = 1/2$ and multiplies the standard NLL target function.

$\textit{I do not know which of the four of these points are important or not. There are still a few recommendations in the four papers}$
$\textit{referenced to improve the variance estimates that could be implemented.}$

$\textit{There is a insideous memory leak in Tensorflow. I need to recompile the models between each cycle as the fixed weights and target}$
$\textit{function changes. Everytime the model recompiles, a new "graph" is created in the backend of Tensorflow and the old graph is not}$
$\textit{released from memory. This is extremely bad when I am training 10 - 20 models per lattice system and I need to recompile the}$
$\textit{model 10 times during the training. I have spent some time trying to fix it without resolution. This is not the first time I have}$
$\textit{struggled with Tensorflow memory leaks and feel like jumping ship to Pytorch might be justified to solve this issue.}$

Each of the networks separately predicting $\mu$, $\alpha$ and $\beta$ are dense networks (MLP, Feed-forward). I do have an LSTM model that I used with orthorhombic. However I am omitting this from the discussion. 

The default mean network uses three hidden layers of 200, 100, and 60 units. These hidden layers proceed in the following order, 1) Dense layer, 2) Layer Normalization, 3) GELU activation, 4) Dropout at a rate of 50% $. The final layer is a Dense layer that outputs with linear activation.

The default $\alpha$ and $\beta$ networks use two hidden layers of 100 and 60 units and are otherwise similar to the mean network. The final layer of the $\alpha$ and $\beta$ networks uses a softplus activation to enforce a positive constraint on $\beta$ and one is added to the $\alpha$ network after softplus activation to enforce an $\alpha > 1$ constraint.

All training occurs with the Adam optimizer using a learning rate of 0.0001 and a batch size of 64. The Neural networks are implemented in Tensorflow.

$\textit{I am looking through the code right now and it looks like the way I am setting the defaults is laden with bugs}$
$\textit{For example, the dropout rate is actually 0 due to a bug. The listed hyperparameters are likely inaccurate}$
$\textit{I have not done any hyperparameter optimization. There is a lot of over and under fitting.}$

### 2.2.3 Random forest
In addition to the neural networks described in Section 2.2.2, random forest regression models are fit to each group of data. These are implemented using SKlearn with the parameters: 80 decision trees each trained with a 10% subsample of the training set and a minimum of 10 samples per leaf. Otherwise the parameters were the defaults of $\textit{sklearn.ensemble.RandomForestRegressor}$.

$\textit{I have not done the evaluation, but I would like to know how similar or different the random forest and neural network models are}$
$\textit{If the predictions overlap 100\%, but the neural networks are more accurate, the random forest models could be dropped}$
$\textit{If the predictions occasionally produce very different estimates, it would make sense to keep both models}$

### 2.2.4 Evaluations

### 2.3 Miller Index Assignment Model
The second primary component of an indexing algorithm is the assignment of Miller indices give a unit cell and observed peak spacings. This is performed by establishing a reference set of Miller indices. These are all the possible Miller indices for a given latttice system without consideration for systematic absences. The forward diffraction models are used to calculate a reference set of peak spacings from this reference set of Miller indices and a given unit cell. A pairwise difference array is then calculated between the observed and reference set of peak spacings. This array is then used as inputs to a neural network which produces a probability distribution, for each observed peak, for the correct Miller index assignment over the reference set of Miller indices.

### 2.3.1 Reference Miller Indices
One reference set of Miller indices is created for each lattice system. These are created by populating an array of all possible Miller indices with $h,k,l$ each ranging between -20 and 20, then reducing this list to Miller indices that give unit peak spacings for any unit cell within the constraints of the lattice system. This Miller index list is then sorted in a manner that roughly should increase with peak spacing.

1) Cubic lattice system
For each Miller index in the "all possible set", $h^2 + k^2 + l^2$ is calculated. Each unique value of $h^2 + k^2 + l^2$ results in a unique peak. These are selected then sorted in order of increasing $h^2 + k^2 + l^2$.

2) Tetragonal lattice system
For each Miller index in the "all possible set", $h^2 + k^2$, and $l^2$ are calculated. Each unique set of $h^2 + k^2$ and $l^2$ results in a unique peak. These are selected then sorted in order of increasing $h^2 + k^2 + l^2$.

3) Hexagonal lattice system
For each Miller index in the "all possible set", $h^2 + hk + k^2$, and $l^2$ are calculated. Each unique set of $h^2 + hk + k^2$, and $l^2$ results in a unique peak. These are selected then sorted in order of increasing $\frac{4}{3}(h^2 + hk + k^2) + l^2$.

4) Rhombohedral lattice system
For each Miller index in the "all possible set", $h^2 + k^2 + l^2$ and $hk + kl + hl$ are calculated. Each unique set of $h^2 + k^2 + l^2$ and $hk + kl + hl$ results in a unique peak. These are selected then sorted in order of increasing $h^2 + k^2 + l^2$.

5) Orthorhombic lattice system
For each Miller index in the "all possible set", $h^2$, $k^2$, and $l^2$ are calculated. Each unique set of $h^2$, $k^2$ and $l^2$ results in a unique peak. These are selected then sorted in order of increasing $h^2 + k^2 + l^2$.

6) Monoclinic lattice system
For each Miller index in the "all possible set", $h^2$, $k^2$, $l^2$, and $hl$ are calculated. Each unique set of $h^2$, $k^2$, $l^2$, and $hl$ results in a unique peak. These are selected then sorted in order of increasing $h^2 + k^2 + l^2$.

7) Triclinic lattice system
For each Miller index in the "all possible set", $h^2$, $k^2$, $l^2$, $hk$, $kl$ and $hl$ are calculated. Each unique set of $h^2$, $k^2$, $l^2$, $hk$, $kl$ and $hl$ results in a unique peak. These are selected then sorted in order of increasing $h^2 + k^2 + l^2$.

One model is created for each lattice system. The last step to establish the reference set of Miller indices is a final sorting. The peak spacing is calculated for each of peak in the set of unaugmented training data given the reference Miller index. The average peak spacings are then averaged over all the entries. Then the reference Miller index set is sorted so these averaged peak spacings increase monotonically. For cubic, a Miller index list of 100 is used, for all other lattice systems 500 is used. The Miller indices taken from the peak list are then assigned a nominal label corresponding to the position of the equivalent Miller index in the sorted reference list. If a peak's Miller index is not found in the reference list, which occurs because the equivalent Miller index is beyond the 100 or 500 length cutoff, it is assigned the last nominal label which is given a Miller index of (000).

### 2.3.3 Neural Network
The inputs of the neural network are scaled unit cell parameters and scaled $\vec{q}^2_{obs}$. Pairwise differences are calculated by first calculating the scaled peak spacing for each reference Miller index given the input unit cell parameter, $\vec{q}^2_{ref}$. This is subtracted from the observed peak spacing in a pairwise manner, $\Delta_{\vec{q}^2} = \vec{q}^2_{obs} - \vec{q}^{2\ T}_{ref}$. If there are 20 input peaks and a reference peak list of 500, this will result in a 20 x 500 array. If the pairwise differences are used to predict peak positions, the correct assignment will occur at small values of $\Delta_{\vec{q}^2}$. Additionally, incorrect assignment will occur at large absolute values of $\Delta_{\vec{q}^2}$. $\Delta_{\vec{q}^2}$ is transformed so it will span the range of 0 to 1, where the correct assignment will occur at larger values,

$\Delta^*_{\vec{q}^2} = \frac{\epsilon_\Delta}{|\Delta_{\vec{q}^2}| + \epsilon_\Delta}$.

$\Delta^*_{\vec{q}^2}$ is used as the input to a dense neural network. The default network uses four hidden layers of 200, 200, 200, and 100 units. These hidden layers proceed in the following order, 1) Dense layer, 2) Layer Normalization, 3) GELU activation, 4) Dropout at a rate of 25%. These hidden layers operate on two-dimensional arrays of shape (# peaks, # reference Miller indices) for the first hidden layer and (# peaks, # previous layer units) for the second hidden layer. The Dense layers in Tensorflow operate on these 2D inputs as follows.

1) Create a kernel with shape (# previous layer units, # current layer units).
2) For each peak dimension, operate on the input array with the same kernel from step 1).

In otherwords, the same kernel operates on each peak dimension in the hidden layer independently. I believe the implications of this are 1) information isn't being shared between peak dimensions, and 2) Each peak dimension is considered in an equivalent manner.

The final layer is a Dense layer that outputs with softmax activation. Instead of one Dense layer output, the network is branched for each peak and a separate dense layer - meaning a separate kernel - acts on the one dimensional input for the given peak.

An important question to address is, what data should these networks be trained with? One option is the true unit cell parameters, this results in 100% accurate Miller index assignments (and it does). On the other hand, they could be trained using the predicted unit cell parameters from the regression networks. This should represent worse case scenario for unit cells. There should be a few randomly generated unit cell better than the expected value and after several rounds of optimization, the best candidate unit cells should be much improved. Ideally, the Miller index assignment networks would be trained on unit cell parameters that differ from the true values by amounts consistent with the current best candidate unit cells during optimization. To these means, a series of assignment networks are trained. The first network is trained using predicted unit cells, where the predictions are performed with the neural network model corresponding to the true group that the entry belongs to. Subsequent networks are trained with randomly perturbed unit cells. This is performed by adding gaussian noise to the scaled unit cell parameters before they are used to calculate pairwise differences. This is setup in a way that each entry receives a different perturbation each training epoch to serve as additional regularization.

Training of the networks is performed with all the unaugmented entries for a given lattice system. Augmented entries are omitted because there is plenty of unaugmented data to train the networks. The Sparse categorical cross-entropy target function is choosen and the Adam optimizer is used with a learning rate of 0.002 and a batch size of 256. The Neural networks are implemented in Tensorflow. Networks are trained with Gaussian noise levels of 0.10, 0.05, 0.025, and 0.01. For monoclinic, the associated $\epsilon_\Delta$ values were 0.01, 0.0075, 0.01, and 0.005.

### 2.3.4 Evaluations

### 2.4 Optimization

The third component of the indexing algorithm is the optimization of unit cell parameters given initial unit cell candidates and a mechanism to assign Miller indices. Currently, the program only considers one lattice system at a time. So if a crystal is known to be in monoclinic, it will be optimized with that as a known fact. After getting this working with all lattice systems, I will work on that aspect.

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

From here, this likelihood function is simplified. The probability distribution of the unit cells lengths is assumed to be constant and bounded between 2 and 500 $\AA$. The probability distribution of the unit cells angles is assumed to be constant and bounded between 0$^o$ and 180$^o$ $\AA$. The term $\rho(\vec{c})$ is removed from the likelihood function and out-of-bounds unit cells are removed from consideration,

$\rho(H,\vec{c}|\vec{q}_{obs}) \propto \Pi_{k=1}^{n_k}\left[\Sigma_{h=0}^{n_h}\rho(q_{obs,k}|H_h, \vec{c})\rho(H_h|\vec{c})\right]$

$\textit{On my to-do list is to optimize this term directly. It is entirely possible with the current set of infrastructure. The }$
$\textit{Miller index neural networks are differentiable. I can optimize the unit cell parameters considering all possible Miller}$
$\textit{Index assigments. Until then, I am only considering one at a time for simplication}$

The likelihood equation is again simplified by only considering the $j^{th}$ Miller index assignment at a time. The most obvious choice is the most probable Miller index assignment, but this does not necessarily the case.

$\rho(H,\vec{c}|\vec{q}_{obs}) \propto \Pi_{k=1}^{n_k}\rho(q_{obs,k}|H_j, \vec{c})\rho(H_j|\vec{c})$.

The term $\rho(H_j|\vec{c})$ is the softmax output from the assignment neural network. The term $\rho(q_{obs,k}|H_j, \vec{c})$ is a normal distribution,

$\rho(q_{obs,k}|H_j, \vec{c}) = \frac{1}{\sqrt{2\pi}\sigma_{q^2}} \exp\left(-\frac{1}{2}\frac{(q^2_{obs,k} - q^2_{cal,k})^2}{\sigma^2_{q^2}}\right)$.

The forward diffraction models are used to calculate $q^2_{cal,k}$. The standard deviation paramter, $\sigma^2_{q^2}$, is defined in a manner similar to the choice of weighting by SVD-Index,

$\sigma^2_{q^2} = q^2_{obs,k} |q^2_{obs,k} - q^2_{cal,k} + \epsilon_{q^2}|$.

Notably, the $\sigma^2_{q^2}$ presented here does not include the randomness in the same way as the weighting in SVD-Index. Randomness is important to the algorithm and is included by other means as described later in the methods.

### 2.4.2 Random Unit Cell Generation and Miller Index Assignments

Random unit cells are generated from the neural networks. For each entry, a set of mean and variance estimates are taken from the each neural networks trained on the different groups. Each set of mean and variance estimates are used as parameters for a Normal distribution and 30 random unit cell parameters are generated. Thirty random unit cells are also selected from from individual decision trees in the random forest model.

Once random unit cells are generated, there will be $n_g(n_{nn} + n_{rf})$ candidate unit cells where $n_g$ is the number of groups and $n_{nn}$, $n_{rf}$ are the number of unit cell candidates generated from each neural network and random forest model respectively.

For each candidate unit cell, Miller indices are assigned to each peak using the neural network model. If any candidates have the same set of Miller index assignments, the unit cell parameters are averaged together to make one single candidate.

### 2.4.3 Unit Cell Optimization

At this point, there are candidate unit cells, assigned Miller indices with confidences $\rho(H_j|\vec{c})$. An novel algorithm within the Markov Chain Monte Carlo (MCMC) framework is used to perform statistical optimization of the unit cell parameters. A MCMC algorithm is a statistical optimization method the roughly follows the steps 0) Start with a initial set of parameters and a likelihood function. 1) select a new set of parameters according to a well defined protocol that includes a randomness aspect. 2) Either accept or reject the new set of parameters based on the ratio of the likelihoods (and/or priors) calculated from the new set and original set of parameters. Then iteratively repeat steps 1) and 2), until a predefined stopping point or criteria.

For this algorithm, MCMC Step 0) is described by section 2.4.2. MCMC Step 1), new unit cell parameters are selected according to this protocol:

1.1) Subsample the peaks, so if there are 20 peaks, 15 peaks are randomly selected. Options exist for purely random subsampling and for subsampling based on softmax probabilities.

$\textit{Subsampling peaks might not work for triclinic. There might not be enough peaks to ensure there is a constraint for each unit cell parameter.}$
$\textit{Well, Coehlo 2003 shows that 20 peaks will result in 12\% of triclinic unit cells will be unsolvable. 9\% for monoclinic}$
$\textit{I plan on implementing a resampling approach where Miller indices are choosen instead of the most probable (largest softmax), the resampling}$
$\textit{approach will assign Miller indices by random selection based on the softmax probabilities.}$

1.2) Optimize the unit cell parameters by minimizing the negative log-likelihood given by Eq. X. This is performed with $\textit{scipy.optimize.minize}$, both analytically calculated gradients and Hessians are available. Notably, the optimization is formulated so the second derivatives of $q^2_{cal}$ are zero with respect to the optimizable parameters, easing the calculation of the Hessian making second derivative based optimization practical, as opposed to Quasi-Newton approaches. The default optimization algorithm is the 'dogleg' method, although other options are available. Immediately before optimization, $\sigma^2_{q^2}$ values are calculated and left stationary during optimization.



### 2.4.4 Found explainers

# 3: Results

# 4: References

Coelho, A. A., (2003) J. Appl. Cryst. 36, 86-95.

Detlefsen, N. S., Jørgensen, M., Hauberg, S. (2019) 33rd Conference on Neural Information Processing Systems (NeurIPS 2019), Vancouver, Canada.

Lafuente B, Downs R T, Yang H, Stone N (2015) The power of databases: the RRUFF project. In: Highlights in Mineralogical Crystallography, T Armbruster and R M Danisi, eds. Berlin, Germany, W. De Gruyter, pp 1-30

Seitzer, M., (2022) arXiv:2203.09168v2

Stirn, A., Knowles, D. A., (2021) arXiv:2006.04910v3

Stirn, A. etal (2022) arXiv:2212.09184v1
        