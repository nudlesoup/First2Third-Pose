# Enhancing Egocentric 3D Pose Estimation with Third Person Views

## Abstract
In this Project, we propose a novel approach to enhance the 3D body pose estimations of a person computed from videos captured from a single wearable camera. 
The key idea is to leverage high-level features linking first- and third-views in a joint embedding space. To learn such embedding space we introduce First2Third, a new paired synchronized dataset of nearly 2,000 videos depicting human activities captured from both a first and a third view perspective. We explicitly consider spatial- and motion-domain features,  combined using a semi-Siamese architecture trained in a self-supervised fashion.

Experimental results demonstrate that the joint multi-view embedded space learned with our dataset is useful to extract discriminatory features from arbitrary single-view egocentric videos, leading to significant improvement of egocentric 3D body pose estimation performance for three supervised state-of-the-art approaches.

Links to Code and Dataset will be updated soon!
### [arXiv] [Code & Data]

## Figure
![alt text](https://github.com/nudlesoup/First2Third/blob/main/SelfSupervisedModel1.1.png?raw=true)


## Results

## Authors
[Ameya Dhamanaskar](https://nudlesoup.github.io/), [Mariella Dimiccoli](https://www.iri.upc.edu/people/mdimiccoli/), [Enric Corona](https://www.iri.upc.edu/people/ecorona/), [Albert Pumarola](https://www.albertpumarola.com/), [Francesc Moreno Noguer](http://www.iri.upc.edu/people/fmoreno/).

## Citing First2Third
If you use First2Third in your research or wish to refer to the baseline results published in the paper, please use the following BibTeX entry.

```BibTeX
@inproceedings{ameya_2022_First2Third,
  author =       {Ameya Dhamanaskar, Mariella Dimiccoli, Enric Corona, 
                  Albert Pumarola, Francesc Moreno-Noguer},
  title =        {Enhancing Egocentric 3D Pose Estimation with Third Person Views},
  year =         {2022}
}
```
