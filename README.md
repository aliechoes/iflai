# Imaging FLow cytometry and AI (iflai)

In this repository, a series of machine learning approaches are implemented to classify and analyze Imaging Flow Cytometry data. 

A significant contribution of IFAI is the focus on reproducibility and interpretability in the analysis part. We have provided an open-source python package using only well-known libraries such as SciPy, NumPy, Pandas, scikit-learn, and scikit-image. This design guarantees that the package can be maintained and updated with minimal effort. This package can be used on multiple CPUs, enabling high-performance computing use-cases. This parallelism is essential as imaging flow cytometry is a high-throughput technology. This package is designed as a standalone implementation that is easier to use compared to the previous work of Henning et al., which depends on other software such as CellProfiler.  IFAI focuses on interpretability without compromising performance, which is different from Deepometry, which focuses on performance by using a ResNet50 as the backbone. This interpretability enables the investigation of different aspects of the cells. 

## Data structure

For using the package, you need to download the data from IDEAS software and save each image (all the channels) as an `.h5` file. The `.h5` file should include at least these keys: `image`, `mask`. In case there is label available, also the `label` should be provided as `str`.

In addition, each file should be saved with the object number as the last part in the name. For example, for a random image with object number of 1000 this is the correct name: `random_file_1000.h5`. This is important as you can use the object numbers to come back to files and use the IDEAS software as well.

Apart from each file, we assume that the data comes from different experiments, donors and conditions. For example, in case we have N experiments, M donors and K conditions, the data path folder should look like this:

```
data_path/Experiment_1/Donor_1/condition_1/*.h5
data_path/Experiment_1/Donor_1/condition_2/*.h5
.
.
.
data_path/Experiment_1/Donor_2/condition_1/*.h5
data_path/Experiment_1/Donor_2/condition_2/*.h5
.
.
.
data_path/Experiment_N/Donor_M/condition_K/*.h5
```
