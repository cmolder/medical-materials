# Learning Medical Materials from Radiography Images
This is the source code for the implementation of "Learning Medical Materials from Radiography Images" (Molder, Lowe, and Zhan). This code serves as a deep learning framework for learning materials in medical radiography images like X-rays and MRIs.



## About

To be written



## Usage

To run our implementation, we provide an [Anaconda](https://www.anaconda.com/) environment. If Anaconda is installed, our environment can be installed by running the following command in the root directory of the repository:

`conda env create --file environment.yml`

### Generating image patches

To be written

### Training and evaluating D-CNN

To be written

### Generating A matrix

To be written

### Training and evaluating MAC-CNN

To be written


## Acknowledgements

To be written

### Data acknowledgements
In our paper, we used the following datasets to generate medical texture patches and evaluate D-CNN and MAC-CNN:

#### Brain / tumor datasets
[Brain-Tumor-Progression, The Cancer Imaging Archive](https://wiki.cancerimagingarchive.net/display/Public/Brain-Tumor-Progression)

> Schmainda KM, Prah M (2018). **Data from Brain-Tumor-Progression.** The Cancer Imaging Archive. http://doi.org/10.7937/K9/TCIA.2018.15quzvnb 

> Clark K, Vendt B, Smith K, Freymann J, Kirby J, Koppel P, Moore S, Phillips S, Maffitt D, Pringle M, Tarbox L, Prior F. **The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository**, Journal of Digital Imaging, Volume 26, Number 6, December, 2013, pp 1045-1057. (paper)

[Brain Tumor Dataset, FigShare](https://search.datacite.org/works/10.6084/M9.FIGSHARE.1512427.V5)

> Cheng, Jun (2017). **brain tumor dataset.** figshare. Dataset. https://doi.org/10.6084/m9.figshare.1512427.v5

#### Bone / implant datasets
[CHECK (Cohort Hip and Cohort Knee) Dataset](https://easy.dans.knaw.nl/ui/datasets/id/easy-dataset:62955), [(link 2)](https://www.check-onderzoek.nl/contact/field-of-interest/)

> Bijlsma, MD, PhD, Professor J.W.J. (University Medical Center Utrecht); Wesseling, PhD J. (University Medical Center Utrecht) (2015): CHECK (Cohort Hip & Cohort Knee) data of baseline (T0). DANS. https://doi.org/10.17026/dans-xs3-ws3s


## Citation

If you use our code or think our work is relevant to yours, we encourage you to cite our paper:

```bibtex
@article{molder2021materials,
    title = {Learning Medical Materials from Radiography Images},
    author = {Molder, Carson and Lowe, Benjamin and Zhan, Justin},
    journal = {Under review},
    year = {2021}
}
```