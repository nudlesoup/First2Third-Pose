## First2Third Dataset

First2Third Dataset contains synchronized frames/videos with Ground truth from Front, Side and Ego-centric view. We also provide Top views for a few of the activities performed in a specific location for 15 people we record.  This is large dataset of short videos covering a variety of human pose types and including multi-third-person-views in addition to first-person view.


Since the orginal dataset is huge. We divide the dataset into 7 parts for easier downloads.  Part 1-6 has total data of 2 people each(every video acvitiy is recorded with a single person) and Part-7 has data for 3 people. Each Part should be around 25 Gbs in size and should contain Egoview, Frontview, Sideview, GT and Topview folders.

The data is currently split into 7 parts which can be downloaded using these links:

 https://cv.iri.upc-csic.es/Dataset/EgoPose/First2Third-Part1.tar.gz

 https://cv.iri.upc-csic.es/Dataset/EgoPose/First2Third-Part2.tar.gz

 https://cv.iri.upc-csic.es/Dataset/EgoPose/First2Third-Part3.tar.gz

 https://cv.iri.upc-csic.es/Dataset/EgoPose/First2Third-Part4.tar.gz

 https://cv.iri.upc-csic.es/Dataset/EgoPose/First2Third-Part5.tar.gz

 https://cv.iri.upc-csic.es/Dataset/EgoPose/First2Third-Part6.tar.gz

 https://cv.iri.upc-csic.es/Dataset/EgoPose/First2Third-Part7.tar.gz


 Alternative temporary link with 7 parts:
 ```
https://www.dropbox.com/sh/lk1rhqzn38zk6ne/AADrIYIaObgAKdrtMkt3aaMTa?dl=0
 ```

 More about Code, [Siamese Model]( https://www.dropbox.com/scl/fi/4nzf6amtch7cn5nr5cpq7/Final-Siamese-Model.ckpt?rlkey=9y48kmttjcvyt284cd638dt8c&dl=0) and how to use organize this data for training/testing COMING SOON!

## Authors
[Ameya Dhamanaskar](https://nudlesoup.github.io/), [Mariella Dimiccoli](https://www.iri.upc.edu/people/mdimiccoli/), [Enric Corona](https://www.iri.upc.edu/people/ecorona/), [Albert Pumarola](https://www.albertpumarola.com/), [Francesc Moreno Noguer](http://www.iri.upc.edu/people/fmoreno/).

### Reference
If you use First2Third dataset in your research or wish to refer to the baseline results published in the paper, please use the following BibTeX entry.

```BibTeX
@article{DHAMANASKAR2023109358,
title = {Enhancing egocentric 3D pose estimation with third person views},
journal = {Pattern Recognition},
volume = {138},
pages = {109358},
year = {2023},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2023.109358},
url = {https://www.sciencedirect.com/science/article/pii/S0031320323000596},
author = {Ameya Dhamanaskar and Mariella Dimiccoli and Enric Corona and Albert Pumarola and Francesc Moreno-Noguer},
keywords = {3D pose estimation, Self-supervised learning, Egocentric vision}
}
```
