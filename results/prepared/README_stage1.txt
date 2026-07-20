Stage 1 data preparation completed
=================================

1. Study scope
--------------

Project:
    MGYS00002437

Environmental classification:
    seawater

Eligible high-depth reference samples:
    474

Minimum high-depth reference sequencing depth:
    10,000 reads


2. Prediction task
------------------

Input:
    Features obtained from simulated shallow 16S sequencing observations.

Target:
    The high-depth observed reference relative abundance for the same
    biological sample and model component.

The target is a high-depth sequencing-derived reference observation.
It is not absolute microbial abundance and is not assumed to be a
noise-free biological truth.


3. Shallow sequencing simulation
---------------------------------

Shallow depth:
    2,000 reads

Subsampling repeats per biological sample:
    5

Simulation method:
    Multinomial sampling from each sample's high-depth reference
    relative-abundance vector.


4. Biological-sample split
--------------------------

Training samples:
    331

Validation samples:
    71

Test samples:
    72

Samples were split before shallow simulations were generated.
All repeats from one biological sample therefore remain in one subset.


5. OTU representation
---------------------

Specific selected OTUs:
    254

Aggregated components:
    1 OTHER component

Total model components:
    255

The OTU vocabulary was selected using training samples only.
All non-selected OTUs were combined into OTHER.


6. Training-only OTU features
-----------------------------

The following OTU-level features were calculated from training samples only:

    otu_mean_ra_train
    otu_prevalence_train
    otu_std_ra_train
    otu_max_ra_train


7. Prepared model datasets
--------------------------

Training rows:
    422,025

Validation rows:
    90,525

Test rows:
    91,800


8. Interpretation limits
------------------------

The analysis may be interpreted as recovery of high-depth observed
relative abundance from shallow 16S observations for new independent
samples from the same seawater study and fixed OTU vocabulary.

The analysis must not be interpreted as:

1. prediction of absolute microbial abundance;
2. prediction of a noise-free biological truth;
3. prediction of completely new habitats;
4. prediction of taxonomic units absent from the training-derived
   vocabulary;
5. evidence that high-depth sequencing perfectly represents the
   underlying community.


9. Next modelling stage
-----------------------

The prepared datasets can be used to compare:

1. raw shallow relative abundance;
2. training-mean relative-abundance baseline;
3. Random Forest regression;
4. XGBoost regression.
