# Enhancing Egocentric 3D Pose Estimation with Third Person Views

## Abstract
We propose a novel approach to enhance the 3D body pose estimation of a person computed from videos captured from a single wearable camera. 
The  main technical contribution consists of leveraging high-level features linking first- and third-views in a joint embedding space. To learn such embedding space we introduce First2Third-Pose, a new paired synchronized dataset of nearly 2,000 videos depicting human activities captured from both first- and third-view perspectives. We explicitly consider spatial- and motion-domain features,  combined using a semi-Siamese architecture trained in a self-supervised fashion.
Experimental results demonstrate that the joint multi-view embedded space learned with our dataset is useful to extract discriminatory features from arbitrary single-view egocentric videos, with no need to perform any sort of domain adaptation or knowledge of camera parameters. An extensive evaluation demonstrates that we achieve significant improvement in egocentric 3D body pose estimation performance on two unconstrained datasets, over three supervised state-of-the-art approaches. Our dataset and code will be available for research purposes.

Links to Code and Dataset will be updated soon!
### [arXiv : ](https://arxiv.org/pdf/2201.02017.pdf) [![report](https://img.shields.io/badge/arXiv-2201.02017-b31b1b.svg)](https://arxiv.org/abs/2201.02017#)
[Code & Data]

## Approach Overview
Our model uses a semi-Siamese architecture to learn to detect if a pair of first- and third-view videos of the First2Third paired source dataset are syncronized or not, by minimizing a contrastive loss green arrows. %Each stream of the semi-Siamese network takes as inputs stacked RGB and optical flow frames.  

This pretext task leads to learn a joint embedding space, where the gap between the first-view and third-view worlds is minimized. The so learned joint embedding space can in principle be leveraged by any supervised method for 3D egopose estimation on a target dataset, without a need for domain adaptation. At both train time brown arrows and test time blue arrows, the semi-Siamese network is used for feature projection onto the learned joint embedded space. z is 64-
dimensional vector, obtained once removed the softmax layer of the Siamese network pre-trained with our dataset.

![alt text](https://github.com/nudlesoup/First2Third/blob/main/SelfSupervisedModel1.1.png?raw=true)

## Dataset Examples
<table>
  <tr>
    <td>Enric lab egoview</td>
     <td>Enric lab frontview</td>
     <td>Enric lab sideview</td>
     <td>Enric lab topview</td>
  </tr>
  <tr>
    <td><img src="https://github.com/nudlesoup/First2Third/blob/main/enric_basketball_ego.gif" width=235 height=250></td>
    <td><img src="https://github.com/nudlesoup/First2Third/blob/main/enric_basketball_front.gif" width=235 height=250></td>
    <td><img src="https://github.com/nudlesoup/First2Third/blob/main/enric_basketball_side.gif" width=235 height=250></td>
    <td><img src="https://github.com/nudlesoup/First2Third/blob/main/enric_basketball_top.gif" width=235 height=250></td>
  </tr>
 </table>
 
 
 <table>
  <tr>
    <td>Shahrukh outdoor egoview</td>
     <td>Shahrukh outdoor frontview</td>
     <td>Shahrukh outdoor sideview</td>
  </tr>
  <tr>
    <td><img src="https://github.com/nudlesoup/First2Third/blob/main/shahrukh_box_ego.gif" width=300 height=250></td>
    <td><img src="https://github.com/nudlesoup/First2Third/blob/main/shahrukh_box_front.gif" width=300 height=250></td>
    <td><img src="https://github.com/nudlesoup/First2Third/blob/main/shahrukh_box_side.gif" width=300 height=250></td>
  </tr>
 </table>
 
## Authors
[Ameya Dhamanaskar](https://nudlesoup.github.io/), [Mariella Dimiccoli](https://www.iri.upc.edu/people/mdimiccoli/), [Enric Corona](https://www.iri.upc.edu/people/ecorona/), [Albert Pumarola](https://www.albertpumarola.com/), [Francesc Moreno Noguer](http://www.iri.upc.edu/people/fmoreno/).

### Reference
If you use First2Third-Pose in your research or wish to refer to the baseline results published in the paper, please use the following BibTeX entry.

```BibTeX
@article{ameya_2022_First2Third-Pose,
  author =       {Ameya Dhamanaskar, Mariella Dimiccoli, Enric Corona, 
                  Albert Pumarola, Francesc Moreno-Noguer},
  title =        {Enhancing Egocentric 3D Pose Estimation with Third Person Views},
  year =         {2022}
}
```
