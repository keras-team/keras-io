##### Copyright 2022 The Emirhan BULUT.
# Planets and Moons Dataset - AI in Space 🌌 🛰 ☄ 🔭
**Author:** [Emirhan BULUT](https://www.linkedin.com/in/artificialintelligencebulut/)<br>
**Date created:** 2022/05/08<br>
**Last modified:** 2022/11/12<br>

**Description:** This software includes the processing of photographic data of planets, dwarf planets and satellites in space (created by converting 3D photos from NASA into 2D) in many different model types (A model of my own was created specifically).

<table class="tfo-notebook-buttons" align="left">

  <td>
    <a target="_blank" href="https://colab.research.google.com/drive/1fFxfj5jGg4AOq-KFzsDpTuFy0hBVipvx?usp=sharing"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/emirhanai"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
  </td>
</table>
<html>


---
## Dataset
The Planets and Moons dataset consists of approximately 8 confirmed planets, 2 dwarf planets, and 1 Earth's moon, the Moon. It includes 11 classes in total.

There are 149 photos of each planet in 3D. All photos are optimized and ready for convolution for AI applications.

Annotations are licensed by Emirhan BULUT under a CC BY-NC 4.0 license. This software and dataset cannot be used for commercial purposes without permission.

Images are listed as licensed CC BY-NC 4.0. The dataset was collected by Emirhan BULUT. The dataset was compiled and collected in accordance with the sensitivity parameters.

## Artificial Intelligence (AI) Software in Space
The software has been prepared with a 100% accuracy, 0.0025 Loss result, superior to most ready-made models.

Emirhan BULUT

Senior Artificial Intelligence Engineer

##**The coding language used:**

`Python 3.9.8`

##**Libraries Used:**

`NumPy`

`Pandas`

`Tensorflow-Keras`

`Glob`

`Seaborn`

`Matplotlib`

`Os`

<img class="fit-picture"
     src="https://raw.githubusercontent.com/emirhanai/Planets-and-Moons-Dataset-AI-in-Space-/main/Planets%20and%20Moons%20Dataset%20-%20AI%20in%20Space%20%F0%9F%8C%8C%20%F0%9F%9B%B0%20%E2%98%84%20%F0%9F%94%AD.png"
     alt="Planets and Moons Dataset - AI in Space 🌌 🛰 ☄ 🔭 - Emirhan BULUT">
     
### **Developer Information:**

Name-Surname: **Emirhan BULUT**

Contact (Email) : **emirhan@isap.solutions**

LinkedIn : **[https://www.linkedin.com/in/artificialintelligencebulut/][LinkedinAccount]**

[LinkedinAccount]: https://www.linkedin.com/in/artificialintelligencebulut/

Kaggle: **[https://www.kaggle.com/emirhanai][Kaggle]**

Official Website: **[https://www.emirhanbulut.com.tr][OfficialWebSite]**

[Kaggle]: https://www.kaggle.com/emirhanai

[OfficialWebSite]: https://www.emirhanbulut.com.tr

## Citations

If you use the Planets and Moons Dataset - AI in Space 🌌 🛰 ☄ 🔭 dataset in your work, please cite it as:

APA-style citation: "Emirhan BULUT. Planets and Moons Dataset - AI in Space 🌌 🛰 ☄ 🔭 : A public dataset for large-scale multi-label and multi-class image classification, 2022. Available from https://github.com/emirhanai/Planets-and-Moons-Dataset-AI-in-Space and https://www.kaggle.com/datasets/emirhanai/planets-and-moons-dataset-ai-in-space".

```
@article{Planets and Moons Dataset - AI in Space 🌌 🛰 ☄ 🔭,
  title={Planets and Moons Dataset - AI in Space 🌌 🛰 ☄ 🔭 : A public dataset for large-scale multi-label and multi-class image classification},
  author={Emirhan BULUT},
  journal={Dataset available from https://github.com/emirhanai/Planets-and-Moons-Dataset-AI-in-Space and https://www.kaggle.com/datasets/emirhanai/planets-and-moons-dataset-ai-in-space},
  year={2022}
}
```

## 💽 Google Drive Mount


```python
from google.colab import drive
drive.mount('/content/drive')
```

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).


## ⬇️ Data Download


```python
!git clone https://github.com/emirhanai/Planets-and-Moons-Dataset-AI-in-Space.git
```

    Cloning into 'Planets-and-Moons-Dataset-AI-in-Space'...
    remote: Enumerating objects: 55, done.[K
    remote: Counting objects: 100% (55/55), done.[K
    remote: Compressing objects: 100% (48/48), done.[K
    remote: Total 55 (delta 18), reused 0 (delta 0), pack-reused 0[K
    Unpacking objects: 100% (55/55), done.



```python
!unzip /content/Planets-and-Moons-Dataset-AI-in-Space/Planets_Moons_Data.zip
```

    Archive:  /content/Planets-and-Moons-Dataset-AI-in-Space/Planets_Moons_Data.zip
       creating: Planets and Moons/
       creating: Planets and Moons/Earth/
      inflating: Planets and Moons/Earth/Earth (1).jpg  
      inflating: Planets and Moons/Earth/Earth (10).jpg  
      inflating: Planets and Moons/Earth/Earth (100).jpg  
      inflating: Planets and Moons/Earth/Earth (101).jpg  
      inflating: Planets and Moons/Earth/Earth (102).jpg  
      inflating: Planets and Moons/Earth/Earth (103).jpg  
      inflating: Planets and Moons/Earth/Earth (104).jpg  
      inflating: Planets and Moons/Earth/Earth (105).jpg  
      inflating: Planets and Moons/Earth/Earth (106).jpg  
      inflating: Planets and Moons/Earth/Earth (107).jpg  
      inflating: Planets and Moons/Earth/Earth (108).jpg  
      inflating: Planets and Moons/Earth/Earth (109).jpg  
      inflating: Planets and Moons/Earth/Earth (11).jpg  
      inflating: Planets and Moons/Earth/Earth (110).jpg  
      inflating: Planets and Moons/Earth/Earth (111).jpg  
      inflating: Planets and Moons/Earth/Earth (112).jpg  
      inflating: Planets and Moons/Earth/Earth (113).jpg  
      inflating: Planets and Moons/Earth/Earth (114).jpg  
      inflating: Planets and Moons/Earth/Earth (115).jpg  
      inflating: Planets and Moons/Earth/Earth (116).jpg  
      inflating: Planets and Moons/Earth/Earth (117).jpg  
      inflating: Planets and Moons/Earth/Earth (118).jpg  
      inflating: Planets and Moons/Earth/Earth (119).jpg  
      inflating: Planets and Moons/Earth/Earth (12).jpg  
      inflating: Planets and Moons/Earth/Earth (120).jpg  
      inflating: Planets and Moons/Earth/Earth (121).jpg  
      inflating: Planets and Moons/Earth/Earth (122).jpg  
      inflating: Planets and Moons/Earth/Earth (123).jpg  
      inflating: Planets and Moons/Earth/Earth (124).jpg  
      inflating: Planets and Moons/Earth/Earth (125).jpg  
      inflating: Planets and Moons/Earth/Earth (126).jpg  
      inflating: Planets and Moons/Earth/Earth (127).jpg  
      inflating: Planets and Moons/Earth/Earth (128).jpg  
      inflating: Planets and Moons/Earth/Earth (129).jpg  
      inflating: Planets and Moons/Earth/Earth (13).jpg  
      inflating: Planets and Moons/Earth/Earth (130).jpg  
      inflating: Planets and Moons/Earth/Earth (131).jpg  
      inflating: Planets and Moons/Earth/Earth (132).jpg  
      inflating: Planets and Moons/Earth/Earth (133).jpg  
      inflating: Planets and Moons/Earth/Earth (134).jpg  
      inflating: Planets and Moons/Earth/Earth (135).jpg  
      inflating: Planets and Moons/Earth/Earth (136).jpg  
      inflating: Planets and Moons/Earth/Earth (137).jpg  
      inflating: Planets and Moons/Earth/Earth (138).jpg  
      inflating: Planets and Moons/Earth/Earth (139).jpg  
      inflating: Planets and Moons/Earth/Earth (14).jpg  
      inflating: Planets and Moons/Earth/Earth (140).jpg  
      inflating: Planets and Moons/Earth/Earth (141).jpg  
      inflating: Planets and Moons/Earth/Earth (142).jpg  
      inflating: Planets and Moons/Earth/Earth (143).jpg  
      inflating: Planets and Moons/Earth/Earth (144).jpg  
      inflating: Planets and Moons/Earth/Earth (145).jpg  
      inflating: Planets and Moons/Earth/Earth (146).jpg  
      inflating: Planets and Moons/Earth/Earth (147).jpg  
      inflating: Planets and Moons/Earth/Earth (148).jpg  
      inflating: Planets and Moons/Earth/Earth (149).jpg  
      inflating: Planets and Moons/Earth/Earth (15).jpg  
      inflating: Planets and Moons/Earth/Earth (16).jpg  
      inflating: Planets and Moons/Earth/Earth (17).jpg  
      inflating: Planets and Moons/Earth/Earth (18).jpg  
      inflating: Planets and Moons/Earth/Earth (19).jpg  
      inflating: Planets and Moons/Earth/Earth (2).jpg  
      inflating: Planets and Moons/Earth/Earth (20).jpg  
      inflating: Planets and Moons/Earth/Earth (21).jpg  
      inflating: Planets and Moons/Earth/Earth (22).jpg  
      inflating: Planets and Moons/Earth/Earth (23).jpg  
      inflating: Planets and Moons/Earth/Earth (24).jpg  
      inflating: Planets and Moons/Earth/Earth (25).jpg  
      inflating: Planets and Moons/Earth/Earth (26).jpg  
      inflating: Planets and Moons/Earth/Earth (27).jpg  
      inflating: Planets and Moons/Earth/Earth (28).jpg  
      inflating: Planets and Moons/Earth/Earth (29).jpg  
      inflating: Planets and Moons/Earth/Earth (3).jpg  
      inflating: Planets and Moons/Earth/Earth (30).jpg  
      inflating: Planets and Moons/Earth/Earth (31).jpg  
      inflating: Planets and Moons/Earth/Earth (32).jpg  
      inflating: Planets and Moons/Earth/Earth (33).jpg  
      inflating: Planets and Moons/Earth/Earth (34).jpg  
      inflating: Planets and Moons/Earth/Earth (35).jpg  
      inflating: Planets and Moons/Earth/Earth (36).jpg  
      inflating: Planets and Moons/Earth/Earth (37).jpg  
      inflating: Planets and Moons/Earth/Earth (38).jpg  
      inflating: Planets and Moons/Earth/Earth (39).jpg  
      inflating: Planets and Moons/Earth/Earth (4).jpg  
      inflating: Planets and Moons/Earth/Earth (40).jpg  
      inflating: Planets and Moons/Earth/Earth (41).jpg  
      inflating: Planets and Moons/Earth/Earth (42).jpg  
      inflating: Planets and Moons/Earth/Earth (43).jpg  
      inflating: Planets and Moons/Earth/Earth (44).jpg  
      inflating: Planets and Moons/Earth/Earth (45).jpg  
      inflating: Planets and Moons/Earth/Earth (46).jpg  
      inflating: Planets and Moons/Earth/Earth (47).jpg  
      inflating: Planets and Moons/Earth/Earth (48).jpg  
      inflating: Planets and Moons/Earth/Earth (49).jpg  
      inflating: Planets and Moons/Earth/Earth (5).jpg  
      inflating: Planets and Moons/Earth/Earth (50).jpg  
      inflating: Planets and Moons/Earth/Earth (51).jpg  
      inflating: Planets and Moons/Earth/Earth (52).jpg  
      inflating: Planets and Moons/Earth/Earth (53).jpg  
      inflating: Planets and Moons/Earth/Earth (54).jpg  
      inflating: Planets and Moons/Earth/Earth (55).jpg  
      inflating: Planets and Moons/Earth/Earth (56).jpg  
      inflating: Planets and Moons/Earth/Earth (57).jpg  
      inflating: Planets and Moons/Earth/Earth (58).jpg  
      inflating: Planets and Moons/Earth/Earth (59).jpg  
      inflating: Planets and Moons/Earth/Earth (6).jpg  
      inflating: Planets and Moons/Earth/Earth (60).jpg  
      inflating: Planets and Moons/Earth/Earth (61).jpg  
      inflating: Planets and Moons/Earth/Earth (62).jpg  
      inflating: Planets and Moons/Earth/Earth (63).jpg  
      inflating: Planets and Moons/Earth/Earth (64).jpg  
      inflating: Planets and Moons/Earth/Earth (65).jpg  
      inflating: Planets and Moons/Earth/Earth (66).jpg  
      inflating: Planets and Moons/Earth/Earth (67).jpg  
      inflating: Planets and Moons/Earth/Earth (68).jpg  
      inflating: Planets and Moons/Earth/Earth (69).jpg  
      inflating: Planets and Moons/Earth/Earth (7).jpg  
      inflating: Planets and Moons/Earth/Earth (70).jpg  
      inflating: Planets and Moons/Earth/Earth (71).jpg  
      inflating: Planets and Moons/Earth/Earth (72).jpg  
      inflating: Planets and Moons/Earth/Earth (73).jpg  
      inflating: Planets and Moons/Earth/Earth (74).jpg  
      inflating: Planets and Moons/Earth/Earth (75).jpg  
      inflating: Planets and Moons/Earth/Earth (76).jpg  
      inflating: Planets and Moons/Earth/Earth (77).jpg  
      inflating: Planets and Moons/Earth/Earth (78).jpg  
      inflating: Planets and Moons/Earth/Earth (79).jpg  
      inflating: Planets and Moons/Earth/Earth (8).jpg  
      inflating: Planets and Moons/Earth/Earth (80).jpg  
      inflating: Planets and Moons/Earth/Earth (81).jpg  
      inflating: Planets and Moons/Earth/Earth (82).jpg  
      inflating: Planets and Moons/Earth/Earth (83).jpg  
      inflating: Planets and Moons/Earth/Earth (84).jpg  
      inflating: Planets and Moons/Earth/Earth (85).jpg  
      inflating: Planets and Moons/Earth/Earth (86).jpg  
      inflating: Planets and Moons/Earth/Earth (87).jpg  
      inflating: Planets and Moons/Earth/Earth (88).jpg  
      inflating: Planets and Moons/Earth/Earth (89).jpg  
      inflating: Planets and Moons/Earth/Earth (9).jpg  
      inflating: Planets and Moons/Earth/Earth (90).jpg  
      inflating: Planets and Moons/Earth/Earth (91).jpg  
      inflating: Planets and Moons/Earth/Earth (92).jpg  
      inflating: Planets and Moons/Earth/Earth (93).jpg  
      inflating: Planets and Moons/Earth/Earth (94).jpg  
      inflating: Planets and Moons/Earth/Earth (95).jpg  
      inflating: Planets and Moons/Earth/Earth (96).jpg  
      inflating: Planets and Moons/Earth/Earth (97).jpg  
      inflating: Planets and Moons/Earth/Earth (98).jpg  
      inflating: Planets and Moons/Earth/Earth (99).jpg  
       creating: Planets and Moons/Jupiter/
      inflating: Planets and Moons/Jupiter/Jupiter (1).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (10).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (100).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (101).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (102).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (103).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (104).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (105).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (106).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (107).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (108).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (109).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (11).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (110).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (111).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (112).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (113).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (114).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (115).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (116).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (117).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (118).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (119).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (12).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (120).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (121).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (122).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (123).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (124).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (125).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (126).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (127).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (128).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (129).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (13).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (130).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (131).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (132).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (133).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (134).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (135).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (136).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (137).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (138).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (139).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (14).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (140).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (141).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (142).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (143).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (144).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (145).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (146).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (147).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (148).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (149).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (15).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (16).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (17).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (18).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (19).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (2).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (20).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (21).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (22).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (23).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (24).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (25).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (26).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (27).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (28).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (29).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (3).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (30).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (31).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (32).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (33).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (34).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (35).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (36).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (37).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (38).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (39).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (4).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (40).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (41).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (42).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (43).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (44).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (45).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (46).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (47).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (48).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (49).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (5).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (50).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (51).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (52).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (53).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (54).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (55).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (56).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (57).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (58).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (59).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (6).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (60).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (61).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (62).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (63).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (64).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (65).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (66).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (67).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (68).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (69).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (7).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (70).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (71).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (72).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (73).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (74).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (75).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (76).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (77).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (78).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (79).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (8).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (80).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (81).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (82).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (83).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (84).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (85).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (86).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (87).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (88).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (89).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (9).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (90).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (91).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (92).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (93).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (94).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (95).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (96).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (97).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (98).jpg  
      inflating: Planets and Moons/Jupiter/Jupiter (99).jpg  
       creating: Planets and Moons/MakeMake/
      inflating: Planets and Moons/MakeMake/Makemake (1).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (150).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (151).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (152).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (153).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (154).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (155).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (156).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (157).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (158).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (159).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (160).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (161).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (162).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (163).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (164).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (165).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (166).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (167).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (168).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (169).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (170).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (171).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (172).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (173).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (174).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (175).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (176).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (177).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (178).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (179).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (180).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (181).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (182).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (183).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (184).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (185).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (186).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (187).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (188).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (189).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (190).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (191).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (192).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (193).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (194).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (195).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (196).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (197).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (198).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (199).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (200).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (201).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (202).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (203).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (204).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (205).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (206).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (207).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (208).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (209).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (210).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (211).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (212).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (213).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (214).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (215).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (216).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (217).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (218).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (219).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (220).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (221).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (222).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (223).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (224).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (225).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (226).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (227).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (228).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (229).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (230).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (231).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (232).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (233).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (234).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (235).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (236).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (237).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (238).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (239).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (240).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (241).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (242).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (243).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (244).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (245).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (246).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (247).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (248).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (249).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (250).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (251).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (252).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (253).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (254).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (255).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (256).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (257).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (258).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (259).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (260).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (261).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (262).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (263).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (264).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (265).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (266).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (267).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (268).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (269).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (270).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (271).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (272).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (273).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (274).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (275).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (276).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (277).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (278).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (279).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (280).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (281).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (282).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (283).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (284).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (285).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (286).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (287).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (288).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (289).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (290).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (291).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (292).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (293).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (294).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (295).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (296).jpg  
      inflating: Planets and Moons/MakeMake/Makemake (297).jpg  
       creating: Planets and Moons/Mars/
      inflating: Planets and Moons/Mars/Mars (1).jpg  
      inflating: Planets and Moons/Mars/Mars (10).jpg  
      inflating: Planets and Moons/Mars/Mars (100).jpg  
      inflating: Planets and Moons/Mars/Mars (101).jpg  
      inflating: Planets and Moons/Mars/Mars (102).jpg  
      inflating: Planets and Moons/Mars/Mars (103).jpg  
      inflating: Planets and Moons/Mars/Mars (104).jpg  
      inflating: Planets and Moons/Mars/Mars (105).jpg  
      inflating: Planets and Moons/Mars/Mars (106).jpg  
      inflating: Planets and Moons/Mars/Mars (107).jpg  
      inflating: Planets and Moons/Mars/Mars (108).jpg  
      inflating: Planets and Moons/Mars/Mars (109).jpg  
      inflating: Planets and Moons/Mars/Mars (11).jpg  
      inflating: Planets and Moons/Mars/Mars (110).jpg  
      inflating: Planets and Moons/Mars/Mars (111).jpg  
      inflating: Planets and Moons/Mars/Mars (112).jpg  
      inflating: Planets and Moons/Mars/Mars (113).jpg  
      inflating: Planets and Moons/Mars/Mars (114).jpg  
      inflating: Planets and Moons/Mars/Mars (115).jpg  
      inflating: Planets and Moons/Mars/Mars (116).jpg  
      inflating: Planets and Moons/Mars/Mars (117).jpg  
      inflating: Planets and Moons/Mars/Mars (118).jpg  
      inflating: Planets and Moons/Mars/Mars (119).jpg  
      inflating: Planets and Moons/Mars/Mars (12).jpg  
      inflating: Planets and Moons/Mars/Mars (120).jpg  
      inflating: Planets and Moons/Mars/Mars (121).jpg  
      inflating: Planets and Moons/Mars/Mars (122).jpg  
      inflating: Planets and Moons/Mars/Mars (123).jpg  
      inflating: Planets and Moons/Mars/Mars (124).jpg  
      inflating: Planets and Moons/Mars/Mars (125).jpg  
      inflating: Planets and Moons/Mars/Mars (126).jpg  
      inflating: Planets and Moons/Mars/Mars (127).jpg  
      inflating: Planets and Moons/Mars/Mars (128).jpg  
      inflating: Planets and Moons/Mars/Mars (129).jpg  
      inflating: Planets and Moons/Mars/Mars (13).jpg  
      inflating: Planets and Moons/Mars/Mars (130).jpg  
      inflating: Planets and Moons/Mars/Mars (131).jpg  
      inflating: Planets and Moons/Mars/Mars (132).jpg  
      inflating: Planets and Moons/Mars/Mars (133).jpg  
      inflating: Planets and Moons/Mars/Mars (134).jpg  
      inflating: Planets and Moons/Mars/Mars (135).jpg  
      inflating: Planets and Moons/Mars/Mars (136).jpg  
      inflating: Planets and Moons/Mars/Mars (137).jpg  
      inflating: Planets and Moons/Mars/Mars (138).jpg  
      inflating: Planets and Moons/Mars/Mars (139).jpg  
      inflating: Planets and Moons/Mars/Mars (14).jpg  
      inflating: Planets and Moons/Mars/Mars (140).jpg  
      inflating: Planets and Moons/Mars/Mars (141).jpg  
      inflating: Planets and Moons/Mars/Mars (142).jpg  
      inflating: Planets and Moons/Mars/Mars (143).jpg  
      inflating: Planets and Moons/Mars/Mars (144).jpg  
      inflating: Planets and Moons/Mars/Mars (145).jpg  
      inflating: Planets and Moons/Mars/Mars (146).jpg  
      inflating: Planets and Moons/Mars/Mars (147).jpg  
      inflating: Planets and Moons/Mars/Mars (148).jpg  
      inflating: Planets and Moons/Mars/Mars (149).jpg  
      inflating: Planets and Moons/Mars/Mars (15).jpg  
      inflating: Planets and Moons/Mars/Mars (16).jpg  
      inflating: Planets and Moons/Mars/Mars (17).jpg  
      inflating: Planets and Moons/Mars/Mars (18).jpg  
      inflating: Planets and Moons/Mars/Mars (19).jpg  
      inflating: Planets and Moons/Mars/Mars (2).jpg  
      inflating: Planets and Moons/Mars/Mars (20).jpg  
      inflating: Planets and Moons/Mars/Mars (21).jpg  
      inflating: Planets and Moons/Mars/Mars (22).jpg  
      inflating: Planets and Moons/Mars/Mars (23).jpg  
      inflating: Planets and Moons/Mars/Mars (24).jpg  
      inflating: Planets and Moons/Mars/Mars (25).jpg  
      inflating: Planets and Moons/Mars/Mars (26).jpg  
      inflating: Planets and Moons/Mars/Mars (27).jpg  
      inflating: Planets and Moons/Mars/Mars (28).jpg  
      inflating: Planets and Moons/Mars/Mars (29).jpg  
      inflating: Planets and Moons/Mars/Mars (3).jpg  
      inflating: Planets and Moons/Mars/Mars (30).jpg  
      inflating: Planets and Moons/Mars/Mars (31).jpg  
      inflating: Planets and Moons/Mars/Mars (32).jpg  
      inflating: Planets and Moons/Mars/Mars (33).jpg  
      inflating: Planets and Moons/Mars/Mars (34).jpg  
      inflating: Planets and Moons/Mars/Mars (35).jpg  
      inflating: Planets and Moons/Mars/Mars (36).jpg  
      inflating: Planets and Moons/Mars/Mars (37).jpg  
      inflating: Planets and Moons/Mars/Mars (38).jpg  
      inflating: Planets and Moons/Mars/Mars (39).jpg  
      inflating: Planets and Moons/Mars/Mars (4).jpg  
      inflating: Planets and Moons/Mars/Mars (40).jpg  
      inflating: Planets and Moons/Mars/Mars (41).jpg  
      inflating: Planets and Moons/Mars/Mars (42).jpg  
      inflating: Planets and Moons/Mars/Mars (43).jpg  
      inflating: Planets and Moons/Mars/Mars (44).jpg  
      inflating: Planets and Moons/Mars/Mars (45).jpg  
      inflating: Planets and Moons/Mars/Mars (46).jpg  
      inflating: Planets and Moons/Mars/Mars (47).jpg  
      inflating: Planets and Moons/Mars/Mars (48).jpg  
      inflating: Planets and Moons/Mars/Mars (49).jpg  
      inflating: Planets and Moons/Mars/Mars (5).jpg  
      inflating: Planets and Moons/Mars/Mars (50).jpg  
      inflating: Planets and Moons/Mars/Mars (51).jpg  
      inflating: Planets and Moons/Mars/Mars (52).jpg  
      inflating: Planets and Moons/Mars/Mars (53).jpg  
      inflating: Planets and Moons/Mars/Mars (54).jpg  
      inflating: Planets and Moons/Mars/Mars (55).jpg  
      inflating: Planets and Moons/Mars/Mars (56).jpg  
      inflating: Planets and Moons/Mars/Mars (57).jpg  
      inflating: Planets and Moons/Mars/Mars (58).jpg  
      inflating: Planets and Moons/Mars/Mars (59).jpg  
      inflating: Planets and Moons/Mars/Mars (6).jpg  
      inflating: Planets and Moons/Mars/Mars (60).jpg  
      inflating: Planets and Moons/Mars/Mars (61).jpg  
      inflating: Planets and Moons/Mars/Mars (62).jpg  
      inflating: Planets and Moons/Mars/Mars (63).jpg  
      inflating: Planets and Moons/Mars/Mars (64).jpg  
      inflating: Planets and Moons/Mars/Mars (65).jpg  
      inflating: Planets and Moons/Mars/Mars (66).jpg  
      inflating: Planets and Moons/Mars/Mars (67).jpg  
      inflating: Planets and Moons/Mars/Mars (68).jpg  
      inflating: Planets and Moons/Mars/Mars (69).jpg  
      inflating: Planets and Moons/Mars/Mars (7).jpg  
      inflating: Planets and Moons/Mars/Mars (70).jpg  
      inflating: Planets and Moons/Mars/Mars (71).jpg  
      inflating: Planets and Moons/Mars/Mars (72).jpg  
      inflating: Planets and Moons/Mars/Mars (73).jpg  
      inflating: Planets and Moons/Mars/Mars (74).jpg  
      inflating: Planets and Moons/Mars/Mars (75).jpg  
      inflating: Planets and Moons/Mars/Mars (76).jpg  
      inflating: Planets and Moons/Mars/Mars (77).jpg  
      inflating: Planets and Moons/Mars/Mars (78).jpg  
      inflating: Planets and Moons/Mars/Mars (79).jpg  
      inflating: Planets and Moons/Mars/Mars (8).jpg  
      inflating: Planets and Moons/Mars/Mars (80).jpg  
      inflating: Planets and Moons/Mars/Mars (81).jpg  
      inflating: Planets and Moons/Mars/Mars (82).jpg  
      inflating: Planets and Moons/Mars/Mars (83).jpg  
      inflating: Planets and Moons/Mars/Mars (84).jpg  
      inflating: Planets and Moons/Mars/Mars (85).jpg  
      inflating: Planets and Moons/Mars/Mars (86).jpg  
      inflating: Planets and Moons/Mars/Mars (87).jpg  
      inflating: Planets and Moons/Mars/Mars (88).jpg  
      inflating: Planets and Moons/Mars/Mars (89).jpg  
      inflating: Planets and Moons/Mars/Mars (9).jpg  
      inflating: Planets and Moons/Mars/Mars (90).jpg  
      inflating: Planets and Moons/Mars/Mars (91).jpg  
      inflating: Planets and Moons/Mars/Mars (92).jpg  
      inflating: Planets and Moons/Mars/Mars (93).jpg  
      inflating: Planets and Moons/Mars/Mars (94).jpg  
      inflating: Planets and Moons/Mars/Mars (95).jpg  
      inflating: Planets and Moons/Mars/Mars (96).jpg  
      inflating: Planets and Moons/Mars/Mars (97).jpg  
      inflating: Planets and Moons/Mars/Mars (98).jpg  
      inflating: Planets and Moons/Mars/Mars (99).jpg  
       creating: Planets and Moons/Mercury/
      inflating: Planets and Moons/Mercury/Mercury (1).jpg  
      inflating: Planets and Moons/Mercury/Mercury (10).jpg  
      inflating: Planets and Moons/Mercury/Mercury (100).jpg  
      inflating: Planets and Moons/Mercury/Mercury (101).jpg  
      inflating: Planets and Moons/Mercury/Mercury (102).jpg  
      inflating: Planets and Moons/Mercury/Mercury (103).jpg  
      inflating: Planets and Moons/Mercury/Mercury (104).jpg  
      inflating: Planets and Moons/Mercury/Mercury (105).jpg  
      inflating: Planets and Moons/Mercury/Mercury (106).jpg  
      inflating: Planets and Moons/Mercury/Mercury (107).jpg  
      inflating: Planets and Moons/Mercury/Mercury (108).jpg  
      inflating: Planets and Moons/Mercury/Mercury (109).jpg  
      inflating: Planets and Moons/Mercury/Mercury (11).jpg  
      inflating: Planets and Moons/Mercury/Mercury (110).jpg  
      inflating: Planets and Moons/Mercury/Mercury (111).jpg  
      inflating: Planets and Moons/Mercury/Mercury (112).jpg  
      inflating: Planets and Moons/Mercury/Mercury (113).jpg  
      inflating: Planets and Moons/Mercury/Mercury (114).jpg  
      inflating: Planets and Moons/Mercury/Mercury (115).jpg  
      inflating: Planets and Moons/Mercury/Mercury (116).jpg  
      inflating: Planets and Moons/Mercury/Mercury (117).jpg  
      inflating: Planets and Moons/Mercury/Mercury (118).jpg  
      inflating: Planets and Moons/Mercury/Mercury (119).jpg  
      inflating: Planets and Moons/Mercury/Mercury (12).jpg  
      inflating: Planets and Moons/Mercury/Mercury (120).jpg  
      inflating: Planets and Moons/Mercury/Mercury (121).jpg  
      inflating: Planets and Moons/Mercury/Mercury (122).jpg  
      inflating: Planets and Moons/Mercury/Mercury (123).jpg  
      inflating: Planets and Moons/Mercury/Mercury (124).jpg  
      inflating: Planets and Moons/Mercury/Mercury (125).jpg  
      inflating: Planets and Moons/Mercury/Mercury (126).jpg  
      inflating: Planets and Moons/Mercury/Mercury (127).jpg  
      inflating: Planets and Moons/Mercury/Mercury (128).jpg  
      inflating: Planets and Moons/Mercury/Mercury (129).jpg  
      inflating: Planets and Moons/Mercury/Mercury (13).jpg  
      inflating: Planets and Moons/Mercury/Mercury (130).jpg  
      inflating: Planets and Moons/Mercury/Mercury (131).jpg  
      inflating: Planets and Moons/Mercury/Mercury (132).jpg  
      inflating: Planets and Moons/Mercury/Mercury (133).jpg  
      inflating: Planets and Moons/Mercury/Mercury (134).jpg  
      inflating: Planets and Moons/Mercury/Mercury (135).jpg  
      inflating: Planets and Moons/Mercury/Mercury (136).jpg  
      inflating: Planets and Moons/Mercury/Mercury (137).jpg  
      inflating: Planets and Moons/Mercury/Mercury (138).jpg  
      inflating: Planets and Moons/Mercury/Mercury (139).jpg  
      inflating: Planets and Moons/Mercury/Mercury (14).jpg  
      inflating: Planets and Moons/Mercury/Mercury (140).jpg  
      inflating: Planets and Moons/Mercury/Mercury (141).jpg  
      inflating: Planets and Moons/Mercury/Mercury (142).jpg  
      inflating: Planets and Moons/Mercury/Mercury (143).jpg  
      inflating: Planets and Moons/Mercury/Mercury (144).jpg  
      inflating: Planets and Moons/Mercury/Mercury (145).jpg  
      inflating: Planets and Moons/Mercury/Mercury (146).jpg  
      inflating: Planets and Moons/Mercury/Mercury (147).jpg  
      inflating: Planets and Moons/Mercury/Mercury (148).jpg  
      inflating: Planets and Moons/Mercury/Mercury (149).jpg  
      inflating: Planets and Moons/Mercury/Mercury (15).jpg  
      inflating: Planets and Moons/Mercury/Mercury (16).jpg  
      inflating: Planets and Moons/Mercury/Mercury (17).jpg  
      inflating: Planets and Moons/Mercury/Mercury (18).jpg  
      inflating: Planets and Moons/Mercury/Mercury (19).jpg  
      inflating: Planets and Moons/Mercury/Mercury (2).jpg  
      inflating: Planets and Moons/Mercury/Mercury (20).jpg  
      inflating: Planets and Moons/Mercury/Mercury (21).jpg  
      inflating: Planets and Moons/Mercury/Mercury (22).jpg  
      inflating: Planets and Moons/Mercury/Mercury (23).jpg  
      inflating: Planets and Moons/Mercury/Mercury (24).jpg  
      inflating: Planets and Moons/Mercury/Mercury (25).jpg  
      inflating: Planets and Moons/Mercury/Mercury (26).jpg  
      inflating: Planets and Moons/Mercury/Mercury (27).jpg  
      inflating: Planets and Moons/Mercury/Mercury (28).jpg  
      inflating: Planets and Moons/Mercury/Mercury (29).jpg  
      inflating: Planets and Moons/Mercury/Mercury (3).jpg  
      inflating: Planets and Moons/Mercury/Mercury (30).jpg  
      inflating: Planets and Moons/Mercury/Mercury (31).jpg  
      inflating: Planets and Moons/Mercury/Mercury (32).jpg  
      inflating: Planets and Moons/Mercury/Mercury (33).jpg  
      inflating: Planets and Moons/Mercury/Mercury (34).jpg  
      inflating: Planets and Moons/Mercury/Mercury (35).jpg  
      inflating: Planets and Moons/Mercury/Mercury (36).jpg  
      inflating: Planets and Moons/Mercury/Mercury (37).jpg  
      inflating: Planets and Moons/Mercury/Mercury (38).jpg  
      inflating: Planets and Moons/Mercury/Mercury (39).jpg  
      inflating: Planets and Moons/Mercury/Mercury (4).jpg  
      inflating: Planets and Moons/Mercury/Mercury (40).jpg  
      inflating: Planets and Moons/Mercury/Mercury (41).jpg  
      inflating: Planets and Moons/Mercury/Mercury (42).jpg  
      inflating: Planets and Moons/Mercury/Mercury (43).jpg  
      inflating: Planets and Moons/Mercury/Mercury (44).jpg  
      inflating: Planets and Moons/Mercury/Mercury (45).jpg  
      inflating: Planets and Moons/Mercury/Mercury (46).jpg  
      inflating: Planets and Moons/Mercury/Mercury (47).jpg  
      inflating: Planets and Moons/Mercury/Mercury (48).jpg  
      inflating: Planets and Moons/Mercury/Mercury (49).jpg  
      inflating: Planets and Moons/Mercury/Mercury (5).jpg  
      inflating: Planets and Moons/Mercury/Mercury (50).jpg  
      inflating: Planets and Moons/Mercury/Mercury (51).jpg  
      inflating: Planets and Moons/Mercury/Mercury (52).jpg  
      inflating: Planets and Moons/Mercury/Mercury (53).jpg  
      inflating: Planets and Moons/Mercury/Mercury (54).jpg  
      inflating: Planets and Moons/Mercury/Mercury (55).jpg  
      inflating: Planets and Moons/Mercury/Mercury (56).jpg  
      inflating: Planets and Moons/Mercury/Mercury (57).jpg  
      inflating: Planets and Moons/Mercury/Mercury (58).jpg  
      inflating: Planets and Moons/Mercury/Mercury (59).jpg  
      inflating: Planets and Moons/Mercury/Mercury (6).jpg  
      inflating: Planets and Moons/Mercury/Mercury (60).jpg  
      inflating: Planets and Moons/Mercury/Mercury (61).jpg  
      inflating: Planets and Moons/Mercury/Mercury (62).jpg  
      inflating: Planets and Moons/Mercury/Mercury (63).jpg  
      inflating: Planets and Moons/Mercury/Mercury (64).jpg  
      inflating: Planets and Moons/Mercury/Mercury (65).jpg  
      inflating: Planets and Moons/Mercury/Mercury (66).jpg  
      inflating: Planets and Moons/Mercury/Mercury (67).jpg  
      inflating: Planets and Moons/Mercury/Mercury (68).jpg  
      inflating: Planets and Moons/Mercury/Mercury (69).jpg  
      inflating: Planets and Moons/Mercury/Mercury (7).jpg  
      inflating: Planets and Moons/Mercury/Mercury (70).jpg  
      inflating: Planets and Moons/Mercury/Mercury (71).jpg  
      inflating: Planets and Moons/Mercury/Mercury (72).jpg  
      inflating: Planets and Moons/Mercury/Mercury (73).jpg  
      inflating: Planets and Moons/Mercury/Mercury (74).jpg  
      inflating: Planets and Moons/Mercury/Mercury (75).jpg  
      inflating: Planets and Moons/Mercury/Mercury (76).jpg  
      inflating: Planets and Moons/Mercury/Mercury (77).jpg  
      inflating: Planets and Moons/Mercury/Mercury (78).jpg  
      inflating: Planets and Moons/Mercury/Mercury (79).jpg  
      inflating: Planets and Moons/Mercury/Mercury (8).jpg  
      inflating: Planets and Moons/Mercury/Mercury (80).jpg  
      inflating: Planets and Moons/Mercury/Mercury (81).jpg  
      inflating: Planets and Moons/Mercury/Mercury (82).jpg  
      inflating: Planets and Moons/Mercury/Mercury (83).jpg  
      inflating: Planets and Moons/Mercury/Mercury (84).jpg  
      inflating: Planets and Moons/Mercury/Mercury (85).jpg  
      inflating: Planets and Moons/Mercury/Mercury (86).jpg  
      inflating: Planets and Moons/Mercury/Mercury (87).jpg  
      inflating: Planets and Moons/Mercury/Mercury (88).jpg  
      inflating: Planets and Moons/Mercury/Mercury (89).jpg  
      inflating: Planets and Moons/Mercury/Mercury (9).jpg  
      inflating: Planets and Moons/Mercury/Mercury (90).jpg  
      inflating: Planets and Moons/Mercury/Mercury (91).jpg  
      inflating: Planets and Moons/Mercury/Mercury (92).jpg  
      inflating: Planets and Moons/Mercury/Mercury (93).jpg  
      inflating: Planets and Moons/Mercury/Mercury (94).jpg  
      inflating: Planets and Moons/Mercury/Mercury (95).jpg  
      inflating: Planets and Moons/Mercury/Mercury (96).jpg  
      inflating: Planets and Moons/Mercury/Mercury (97).jpg  
      inflating: Planets and Moons/Mercury/Mercury (98).jpg  
      inflating: Planets and Moons/Mercury/Mercury (99).jpg  
       creating: Planets and Moons/Moon/
      inflating: Planets and Moons/Moon/Moon (1).jpg  
      inflating: Planets and Moons/Moon/Moon (10).jpg  
      inflating: Planets and Moons/Moon/Moon (100).jpg  
      inflating: Planets and Moons/Moon/Moon (101).jpg  
      inflating: Planets and Moons/Moon/Moon (102).jpg  
      inflating: Planets and Moons/Moon/Moon (103).jpg  
      inflating: Planets and Moons/Moon/Moon (104).jpg  
      inflating: Planets and Moons/Moon/Moon (105).jpg  
      inflating: Planets and Moons/Moon/Moon (106).jpg  
      inflating: Planets and Moons/Moon/Moon (107).jpg  
      inflating: Planets and Moons/Moon/Moon (108).jpg  
      inflating: Planets and Moons/Moon/Moon (109).jpg  
      inflating: Planets and Moons/Moon/Moon (11).jpg  
      inflating: Planets and Moons/Moon/Moon (110).jpg  
      inflating: Planets and Moons/Moon/Moon (111).jpg  
      inflating: Planets and Moons/Moon/Moon (112).jpg  
      inflating: Planets and Moons/Moon/Moon (113).jpg  
      inflating: Planets and Moons/Moon/Moon (114).jpg  
      inflating: Planets and Moons/Moon/Moon (115).jpg  
      inflating: Planets and Moons/Moon/Moon (116).jpg  
      inflating: Planets and Moons/Moon/Moon (117).jpg  
      inflating: Planets and Moons/Moon/Moon (118).jpg  
      inflating: Planets and Moons/Moon/Moon (119).jpg  
      inflating: Planets and Moons/Moon/Moon (12).jpg  
      inflating: Planets and Moons/Moon/Moon (120).jpg  
      inflating: Planets and Moons/Moon/Moon (121).jpg  
      inflating: Planets and Moons/Moon/Moon (122).jpg  
      inflating: Planets and Moons/Moon/Moon (123).jpg  
      inflating: Planets and Moons/Moon/Moon (124).jpg  
      inflating: Planets and Moons/Moon/Moon (125).jpg  
      inflating: Planets and Moons/Moon/Moon (126).jpg  
      inflating: Planets and Moons/Moon/Moon (127).jpg  
      inflating: Planets and Moons/Moon/Moon (128).jpg  
      inflating: Planets and Moons/Moon/Moon (129).jpg  
      inflating: Planets and Moons/Moon/Moon (13).jpg  
      inflating: Planets and Moons/Moon/Moon (130).jpg  
      inflating: Planets and Moons/Moon/Moon (131).jpg  
      inflating: Planets and Moons/Moon/Moon (132).jpg  
      inflating: Planets and Moons/Moon/Moon (133).jpg  
      inflating: Planets and Moons/Moon/Moon (134).jpg  
      inflating: Planets and Moons/Moon/Moon (135).jpg  
      inflating: Planets and Moons/Moon/Moon (136).jpg  
      inflating: Planets and Moons/Moon/Moon (137).jpg  
      inflating: Planets and Moons/Moon/Moon (138).jpg  
      inflating: Planets and Moons/Moon/Moon (139).jpg  
      inflating: Planets and Moons/Moon/Moon (14).jpg  
      inflating: Planets and Moons/Moon/Moon (140).jpg  
      inflating: Planets and Moons/Moon/Moon (141).jpg  
      inflating: Planets and Moons/Moon/Moon (142).jpg  
      inflating: Planets and Moons/Moon/Moon (143).jpg  
      inflating: Planets and Moons/Moon/Moon (144).jpg  
      inflating: Planets and Moons/Moon/Moon (145).jpg  
      inflating: Planets and Moons/Moon/Moon (146).jpg  
      inflating: Planets and Moons/Moon/Moon (147).jpg  
      inflating: Planets and Moons/Moon/Moon (148).jpg  
      inflating: Planets and Moons/Moon/Moon (15).jpg  
      inflating: Planets and Moons/Moon/Moon (16).jpg  
      inflating: Planets and Moons/Moon/Moon (17).jpg  
      inflating: Planets and Moons/Moon/Moon (18).jpg  
      inflating: Planets and Moons/Moon/Moon (19).jpg  
      inflating: Planets and Moons/Moon/Moon (2).jpg  
      inflating: Planets and Moons/Moon/Moon (20).jpg  
      inflating: Planets and Moons/Moon/Moon (21).jpg  
      inflating: Planets and Moons/Moon/Moon (22).jpg  
      inflating: Planets and Moons/Moon/Moon (23).jpg  
      inflating: Planets and Moons/Moon/Moon (24).jpg  
      inflating: Planets and Moons/Moon/Moon (25).jpg  
      inflating: Planets and Moons/Moon/Moon (26).jpg  
      inflating: Planets and Moons/Moon/Moon (27).jpg  
      inflating: Planets and Moons/Moon/Moon (28).jpg  
      inflating: Planets and Moons/Moon/Moon (29).jpg  
      inflating: Planets and Moons/Moon/Moon (3).jpg  
      inflating: Planets and Moons/Moon/Moon (30).jpg  
      inflating: Planets and Moons/Moon/Moon (31).jpg  
      inflating: Planets and Moons/Moon/Moon (32).jpg  
      inflating: Planets and Moons/Moon/Moon (33).jpg  
      inflating: Planets and Moons/Moon/Moon (34).jpg  
      inflating: Planets and Moons/Moon/Moon (35).jpg  
      inflating: Planets and Moons/Moon/Moon (36).jpg  
      inflating: Planets and Moons/Moon/Moon (37).jpg  
      inflating: Planets and Moons/Moon/Moon (38).jpg  
      inflating: Planets and Moons/Moon/Moon (39).jpg  
      inflating: Planets and Moons/Moon/Moon (4).jpg  
      inflating: Planets and Moons/Moon/Moon (40).jpg  
      inflating: Planets and Moons/Moon/Moon (41).jpg  
      inflating: Planets and Moons/Moon/Moon (42).jpg  
      inflating: Planets and Moons/Moon/Moon (43).jpg  
      inflating: Planets and Moons/Moon/Moon (44).jpg  
      inflating: Planets and Moons/Moon/Moon (45).jpg  
      inflating: Planets and Moons/Moon/Moon (46).jpg  
      inflating: Planets and Moons/Moon/Moon (47).jpg  
      inflating: Planets and Moons/Moon/Moon (48).jpg  
      inflating: Planets and Moons/Moon/Moon (49).jpg  
      inflating: Planets and Moons/Moon/Moon (5).jpg  
      inflating: Planets and Moons/Moon/Moon (50).jpg  
      inflating: Planets and Moons/Moon/Moon (51).jpg  
      inflating: Planets and Moons/Moon/Moon (52).jpg  
      inflating: Planets and Moons/Moon/Moon (53).jpg  
      inflating: Planets and Moons/Moon/Moon (54).jpg  
      inflating: Planets and Moons/Moon/Moon (55).jpg  
      inflating: Planets and Moons/Moon/Moon (56).jpg  
      inflating: Planets and Moons/Moon/Moon (57).jpg  
      inflating: Planets and Moons/Moon/Moon (58).jpg  
      inflating: Planets and Moons/Moon/Moon (59).jpg  
      inflating: Planets and Moons/Moon/Moon (6).jpg  
      inflating: Planets and Moons/Moon/Moon (60).jpg  
      inflating: Planets and Moons/Moon/Moon (61).jpg  
      inflating: Planets and Moons/Moon/Moon (62).jpg  
      inflating: Planets and Moons/Moon/Moon (63).jpg  
      inflating: Planets and Moons/Moon/Moon (64).jpg  
      inflating: Planets and Moons/Moon/Moon (65).jpg  
      inflating: Planets and Moons/Moon/Moon (66).jpg  
      inflating: Planets and Moons/Moon/Moon (67).jpg  
      inflating: Planets and Moons/Moon/Moon (68).jpg  
      inflating: Planets and Moons/Moon/Moon (69).jpg  
      inflating: Planets and Moons/Moon/Moon (7).jpg  
      inflating: Planets and Moons/Moon/Moon (70).jpg  
      inflating: Planets and Moons/Moon/Moon (71).jpg  
      inflating: Planets and Moons/Moon/Moon (72).jpg  
      inflating: Planets and Moons/Moon/Moon (73).jpg  
      inflating: Planets and Moons/Moon/Moon (74).jpg  
      inflating: Planets and Moons/Moon/Moon (75).jpg  
      inflating: Planets and Moons/Moon/Moon (76).jpg  
      inflating: Planets and Moons/Moon/Moon (77).jpg  
      inflating: Planets and Moons/Moon/Moon (78).jpg  
      inflating: Planets and Moons/Moon/Moon (79).jpg  
      inflating: Planets and Moons/Moon/Moon (8).jpg  
      inflating: Planets and Moons/Moon/Moon (80).jpg  
      inflating: Planets and Moons/Moon/Moon (81).jpg  
      inflating: Planets and Moons/Moon/Moon (82).jpg  
      inflating: Planets and Moons/Moon/Moon (83).jpg  
      inflating: Planets and Moons/Moon/Moon (84).jpg  
      inflating: Planets and Moons/Moon/Moon (85).jpg  
      inflating: Planets and Moons/Moon/Moon (86).jpg  
      inflating: Planets and Moons/Moon/Moon (87).jpg  
      inflating: Planets and Moons/Moon/Moon (88).jpg  
      inflating: Planets and Moons/Moon/Moon (89).jpg  
      inflating: Planets and Moons/Moon/Moon (9).jpg  
      inflating: Planets and Moons/Moon/Moon (90).jpg  
      inflating: Planets and Moons/Moon/Moon (91).jpg  
      inflating: Planets and Moons/Moon/Moon (92).jpg  
      inflating: Planets and Moons/Moon/Moon (93).jpg  
      inflating: Planets and Moons/Moon/Moon (94).jpg  
      inflating: Planets and Moons/Moon/Moon (95).jpg  
      inflating: Planets and Moons/Moon/Moon (96).jpg  
      inflating: Planets and Moons/Moon/Moon (97).jpg  
      inflating: Planets and Moons/Moon/Moon (98).jpg  
      inflating: Planets and Moons/Moon/Moon (99).jpg  
       creating: Planets and Moons/Neptune/
      inflating: Planets and Moons/Neptune/Neptune (1).jpg  
      inflating: Planets and Moons/Neptune/Neptune (10).jpg  
      inflating: Planets and Moons/Neptune/Neptune (100).jpg  
      inflating: Planets and Moons/Neptune/Neptune (101).jpg  
      inflating: Planets and Moons/Neptune/Neptune (102).jpg  
      inflating: Planets and Moons/Neptune/Neptune (103).jpg  
      inflating: Planets and Moons/Neptune/Neptune (104).jpg  
      inflating: Planets and Moons/Neptune/Neptune (105).jpg  
      inflating: Planets and Moons/Neptune/Neptune (106).jpg  
      inflating: Planets and Moons/Neptune/Neptune (107).jpg  
      inflating: Planets and Moons/Neptune/Neptune (108).jpg  
      inflating: Planets and Moons/Neptune/Neptune (109).jpg  
      inflating: Planets and Moons/Neptune/Neptune (11).jpg  
      inflating: Planets and Moons/Neptune/Neptune (110).jpg  
      inflating: Planets and Moons/Neptune/Neptune (111).jpg  
      inflating: Planets and Moons/Neptune/Neptune (112).jpg  
      inflating: Planets and Moons/Neptune/Neptune (113).jpg  
      inflating: Planets and Moons/Neptune/Neptune (114).jpg  
      inflating: Planets and Moons/Neptune/Neptune (115).jpg  
      inflating: Planets and Moons/Neptune/Neptune (116).jpg  
      inflating: Planets and Moons/Neptune/Neptune (117).jpg  
      inflating: Planets and Moons/Neptune/Neptune (118).jpg  
      inflating: Planets and Moons/Neptune/Neptune (119).jpg  
      inflating: Planets and Moons/Neptune/Neptune (12).jpg  
      inflating: Planets and Moons/Neptune/Neptune (120).jpg  
      inflating: Planets and Moons/Neptune/Neptune (121).jpg  
      inflating: Planets and Moons/Neptune/Neptune (122).jpg  
      inflating: Planets and Moons/Neptune/Neptune (123).jpg  
      inflating: Planets and Moons/Neptune/Neptune (124).jpg  
      inflating: Planets and Moons/Neptune/Neptune (125).jpg  
      inflating: Planets and Moons/Neptune/Neptune (126).jpg  
      inflating: Planets and Moons/Neptune/Neptune (127).jpg  
      inflating: Planets and Moons/Neptune/Neptune (128).jpg  
      inflating: Planets and Moons/Neptune/Neptune (129).jpg  
      inflating: Planets and Moons/Neptune/Neptune (13).jpg  
      inflating: Planets and Moons/Neptune/Neptune (130).jpg  
      inflating: Planets and Moons/Neptune/Neptune (131).jpg  
      inflating: Planets and Moons/Neptune/Neptune (132).jpg  
      inflating: Planets and Moons/Neptune/Neptune (133).jpg  
      inflating: Planets and Moons/Neptune/Neptune (134).jpg  
      inflating: Planets and Moons/Neptune/Neptune (135).jpg  
      inflating: Planets and Moons/Neptune/Neptune (136).jpg  
      inflating: Planets and Moons/Neptune/Neptune (137).jpg  
      inflating: Planets and Moons/Neptune/Neptune (138).jpg  
      inflating: Planets and Moons/Neptune/Neptune (139).jpg  
      inflating: Planets and Moons/Neptune/Neptune (14).jpg  
      inflating: Planets and Moons/Neptune/Neptune (140).jpg  
      inflating: Planets and Moons/Neptune/Neptune (141).jpg  
      inflating: Planets and Moons/Neptune/Neptune (142).jpg  
      inflating: Planets and Moons/Neptune/Neptune (143).jpg  
      inflating: Planets and Moons/Neptune/Neptune (144).jpg  
      inflating: Planets and Moons/Neptune/Neptune (145).jpg  
      inflating: Planets and Moons/Neptune/Neptune (146).jpg  
      inflating: Planets and Moons/Neptune/Neptune (147).jpg  
      inflating: Planets and Moons/Neptune/Neptune (148).jpg  
      inflating: Planets and Moons/Neptune/Neptune (149).jpg  
      inflating: Planets and Moons/Neptune/Neptune (15).jpg  
      inflating: Planets and Moons/Neptune/Neptune (16).jpg  
      inflating: Planets and Moons/Neptune/Neptune (17).jpg  
      inflating: Planets and Moons/Neptune/Neptune (18).jpg  
      inflating: Planets and Moons/Neptune/Neptune (19).jpg  
      inflating: Planets and Moons/Neptune/Neptune (2).jpg  
      inflating: Planets and Moons/Neptune/Neptune (20).jpg  
      inflating: Planets and Moons/Neptune/Neptune (21).jpg  
      inflating: Planets and Moons/Neptune/Neptune (22).jpg  
      inflating: Planets and Moons/Neptune/Neptune (23).jpg  
      inflating: Planets and Moons/Neptune/Neptune (24).jpg  
      inflating: Planets and Moons/Neptune/Neptune (25).jpg  
      inflating: Planets and Moons/Neptune/Neptune (26).jpg  
      inflating: Planets and Moons/Neptune/Neptune (27).jpg  
      inflating: Planets and Moons/Neptune/Neptune (28).jpg  
      inflating: Planets and Moons/Neptune/Neptune (29).jpg  
      inflating: Planets and Moons/Neptune/Neptune (3).jpg  
      inflating: Planets and Moons/Neptune/Neptune (30).jpg  
      inflating: Planets and Moons/Neptune/Neptune (31).jpg  
      inflating: Planets and Moons/Neptune/Neptune (32).jpg  
      inflating: Planets and Moons/Neptune/Neptune (33).jpg  
      inflating: Planets and Moons/Neptune/Neptune (34).jpg  
      inflating: Planets and Moons/Neptune/Neptune (35).jpg  
      inflating: Planets and Moons/Neptune/Neptune (36).jpg  
      inflating: Planets and Moons/Neptune/Neptune (37).jpg  
      inflating: Planets and Moons/Neptune/Neptune (38).jpg  
      inflating: Planets and Moons/Neptune/Neptune (39).jpg  
      inflating: Planets and Moons/Neptune/Neptune (4).jpg  
      inflating: Planets and Moons/Neptune/Neptune (40).jpg  
      inflating: Planets and Moons/Neptune/Neptune (41).jpg  
      inflating: Planets and Moons/Neptune/Neptune (42).jpg  
      inflating: Planets and Moons/Neptune/Neptune (43).jpg  
      inflating: Planets and Moons/Neptune/Neptune (44).jpg  
      inflating: Planets and Moons/Neptune/Neptune (45).jpg  
      inflating: Planets and Moons/Neptune/Neptune (46).jpg  
      inflating: Planets and Moons/Neptune/Neptune (47).jpg  
      inflating: Planets and Moons/Neptune/Neptune (48).jpg  
      inflating: Planets and Moons/Neptune/Neptune (49).jpg  
      inflating: Planets and Moons/Neptune/Neptune (5).jpg  
      inflating: Planets and Moons/Neptune/Neptune (50).jpg  
      inflating: Planets and Moons/Neptune/Neptune (51).jpg  
      inflating: Planets and Moons/Neptune/Neptune (52).jpg  
      inflating: Planets and Moons/Neptune/Neptune (53).jpg  
      inflating: Planets and Moons/Neptune/Neptune (54).jpg  
      inflating: Planets and Moons/Neptune/Neptune (55).jpg  
      inflating: Planets and Moons/Neptune/Neptune (56).jpg  
      inflating: Planets and Moons/Neptune/Neptune (57).jpg  
      inflating: Planets and Moons/Neptune/Neptune (58).jpg  
      inflating: Planets and Moons/Neptune/Neptune (59).jpg  
      inflating: Planets and Moons/Neptune/Neptune (6).jpg  
      inflating: Planets and Moons/Neptune/Neptune (60).jpg  
      inflating: Planets and Moons/Neptune/Neptune (61).jpg  
      inflating: Planets and Moons/Neptune/Neptune (62).jpg  
      inflating: Planets and Moons/Neptune/Neptune (63).jpg  
      inflating: Planets and Moons/Neptune/Neptune (64).jpg  
      inflating: Planets and Moons/Neptune/Neptune (65).jpg  
      inflating: Planets and Moons/Neptune/Neptune (66).jpg  
      inflating: Planets and Moons/Neptune/Neptune (67).jpg  
      inflating: Planets and Moons/Neptune/Neptune (68).jpg  
      inflating: Planets and Moons/Neptune/Neptune (69).jpg  
      inflating: Planets and Moons/Neptune/Neptune (7).jpg  
      inflating: Planets and Moons/Neptune/Neptune (70).jpg  
      inflating: Planets and Moons/Neptune/Neptune (71).jpg  
      inflating: Planets and Moons/Neptune/Neptune (72).jpg  
      inflating: Planets and Moons/Neptune/Neptune (73).jpg  
      inflating: Planets and Moons/Neptune/Neptune (74).jpg  
      inflating: Planets and Moons/Neptune/Neptune (75).jpg  
      inflating: Planets and Moons/Neptune/Neptune (76).jpg  
      inflating: Planets and Moons/Neptune/Neptune (77).jpg  
      inflating: Planets and Moons/Neptune/Neptune (78).jpg  
      inflating: Planets and Moons/Neptune/Neptune (79).jpg  
      inflating: Planets and Moons/Neptune/Neptune (8).jpg  
      inflating: Planets and Moons/Neptune/Neptune (80).jpg  
      inflating: Planets and Moons/Neptune/Neptune (81).jpg  
      inflating: Planets and Moons/Neptune/Neptune (82).jpg  
      inflating: Planets and Moons/Neptune/Neptune (83).jpg  
      inflating: Planets and Moons/Neptune/Neptune (84).jpg  
      inflating: Planets and Moons/Neptune/Neptune (85).jpg  
      inflating: Planets and Moons/Neptune/Neptune (86).jpg  
      inflating: Planets and Moons/Neptune/Neptune (87).jpg  
      inflating: Planets and Moons/Neptune/Neptune (88).jpg  
      inflating: Planets and Moons/Neptune/Neptune (89).jpg  
      inflating: Planets and Moons/Neptune/Neptune (9).jpg  
      inflating: Planets and Moons/Neptune/Neptune (90).jpg  
      inflating: Planets and Moons/Neptune/Neptune (91).jpg  
      inflating: Planets and Moons/Neptune/Neptune (92).jpg  
      inflating: Planets and Moons/Neptune/Neptune (93).jpg  
      inflating: Planets and Moons/Neptune/Neptune (94).jpg  
      inflating: Planets and Moons/Neptune/Neptune (95).jpg  
      inflating: Planets and Moons/Neptune/Neptune (96).jpg  
      inflating: Planets and Moons/Neptune/Neptune (97).jpg  
      inflating: Planets and Moons/Neptune/Neptune (98).jpg  
      inflating: Planets and Moons/Neptune/Neptune (99).jpg  
       creating: Planets and Moons/Pluto/
      inflating: Planets and Moons/Pluto/Pluto (1).jpg  
      inflating: Planets and Moons/Pluto/Pluto (10).jpg  
      inflating: Planets and Moons/Pluto/Pluto (100).jpg  
      inflating: Planets and Moons/Pluto/Pluto (101).jpg  
      inflating: Planets and Moons/Pluto/Pluto (102).jpg  
      inflating: Planets and Moons/Pluto/Pluto (103).jpg  
      inflating: Planets and Moons/Pluto/Pluto (104).jpg  
      inflating: Planets and Moons/Pluto/Pluto (105).jpg  
      inflating: Planets and Moons/Pluto/Pluto (106).jpg  
      inflating: Planets and Moons/Pluto/Pluto (107).jpg  
      inflating: Planets and Moons/Pluto/Pluto (108).jpg  
      inflating: Planets and Moons/Pluto/Pluto (109).jpg  
      inflating: Planets and Moons/Pluto/Pluto (11).jpg  
      inflating: Planets and Moons/Pluto/Pluto (110).jpg  
      inflating: Planets and Moons/Pluto/Pluto (111).jpg  
      inflating: Planets and Moons/Pluto/Pluto (112).jpg  
      inflating: Planets and Moons/Pluto/Pluto (113).jpg  
      inflating: Planets and Moons/Pluto/Pluto (114).jpg  
      inflating: Planets and Moons/Pluto/Pluto (115).jpg  
      inflating: Planets and Moons/Pluto/Pluto (116).jpg  
      inflating: Planets and Moons/Pluto/Pluto (117).jpg  
      inflating: Planets and Moons/Pluto/Pluto (118).jpg  
      inflating: Planets and Moons/Pluto/Pluto (119).jpg  
      inflating: Planets and Moons/Pluto/Pluto (12).jpg  
      inflating: Planets and Moons/Pluto/Pluto (120).jpg  
      inflating: Planets and Moons/Pluto/Pluto (121).jpg  
      inflating: Planets and Moons/Pluto/Pluto (122).jpg  
      inflating: Planets and Moons/Pluto/Pluto (123).jpg  
      inflating: Planets and Moons/Pluto/Pluto (124).jpg  
      inflating: Planets and Moons/Pluto/Pluto (125).jpg  
      inflating: Planets and Moons/Pluto/Pluto (126).jpg  
      inflating: Planets and Moons/Pluto/Pluto (127).jpg  
      inflating: Planets and Moons/Pluto/Pluto (128).jpg  
      inflating: Planets and Moons/Pluto/Pluto (129).jpg  
      inflating: Planets and Moons/Pluto/Pluto (13).jpg  
      inflating: Planets and Moons/Pluto/Pluto (130).jpg  
      inflating: Planets and Moons/Pluto/Pluto (131).jpg  
      inflating: Planets and Moons/Pluto/Pluto (132).jpg  
      inflating: Planets and Moons/Pluto/Pluto (133).jpg  
      inflating: Planets and Moons/Pluto/Pluto (134).jpg  
      inflating: Planets and Moons/Pluto/Pluto (135).jpg  
      inflating: Planets and Moons/Pluto/Pluto (136).jpg  
      inflating: Planets and Moons/Pluto/Pluto (137).jpg  
      inflating: Planets and Moons/Pluto/Pluto (138).jpg  
      inflating: Planets and Moons/Pluto/Pluto (139).jpg  
      inflating: Planets and Moons/Pluto/Pluto (14).jpg  
      inflating: Planets and Moons/Pluto/Pluto (140).jpg  
      inflating: Planets and Moons/Pluto/Pluto (141).jpg  
      inflating: Planets and Moons/Pluto/Pluto (142).jpg  
      inflating: Planets and Moons/Pluto/Pluto (143).jpg  
      inflating: Planets and Moons/Pluto/Pluto (144).jpg  
      inflating: Planets and Moons/Pluto/Pluto (145).jpg  
      inflating: Planets and Moons/Pluto/Pluto (146).jpg  
      inflating: Planets and Moons/Pluto/Pluto (147).jpg  
      inflating: Planets and Moons/Pluto/Pluto (148).jpg  
      inflating: Planets and Moons/Pluto/Pluto (149).jpg  
      inflating: Planets and Moons/Pluto/Pluto (15).jpg  
      inflating: Planets and Moons/Pluto/Pluto (16).jpg  
      inflating: Planets and Moons/Pluto/Pluto (17).jpg  
      inflating: Planets and Moons/Pluto/Pluto (18).jpg  
      inflating: Planets and Moons/Pluto/Pluto (19).jpg  
      inflating: Planets and Moons/Pluto/Pluto (2).jpg  
      inflating: Planets and Moons/Pluto/Pluto (20).jpg  
      inflating: Planets and Moons/Pluto/Pluto (21).jpg  
      inflating: Planets and Moons/Pluto/Pluto (22).jpg  
      inflating: Planets and Moons/Pluto/Pluto (23).jpg  
      inflating: Planets and Moons/Pluto/Pluto (24).jpg  
      inflating: Planets and Moons/Pluto/Pluto (25).jpg  
      inflating: Planets and Moons/Pluto/Pluto (26).jpg  
      inflating: Planets and Moons/Pluto/Pluto (27).jpg  
      inflating: Planets and Moons/Pluto/Pluto (28).jpg  
      inflating: Planets and Moons/Pluto/Pluto (29).jpg  
      inflating: Planets and Moons/Pluto/Pluto (3).jpg  
      inflating: Planets and Moons/Pluto/Pluto (30).jpg  
      inflating: Planets and Moons/Pluto/Pluto (31).jpg  
      inflating: Planets and Moons/Pluto/Pluto (32).jpg  
      inflating: Planets and Moons/Pluto/Pluto (33).jpg  
      inflating: Planets and Moons/Pluto/Pluto (34).jpg  
      inflating: Planets and Moons/Pluto/Pluto (35).jpg  
      inflating: Planets and Moons/Pluto/Pluto (36).jpg  
      inflating: Planets and Moons/Pluto/Pluto (37).jpg  
      inflating: Planets and Moons/Pluto/Pluto (38).jpg  
      inflating: Planets and Moons/Pluto/Pluto (39).jpg  
      inflating: Planets and Moons/Pluto/Pluto (4).jpg  
      inflating: Planets and Moons/Pluto/Pluto (40).jpg  
      inflating: Planets and Moons/Pluto/Pluto (41).jpg  
      inflating: Planets and Moons/Pluto/Pluto (42).jpg  
      inflating: Planets and Moons/Pluto/Pluto (43).jpg  
      inflating: Planets and Moons/Pluto/Pluto (44).jpg  
      inflating: Planets and Moons/Pluto/Pluto (45).jpg  
      inflating: Planets and Moons/Pluto/Pluto (46).jpg  
      inflating: Planets and Moons/Pluto/Pluto (47).jpg  
      inflating: Planets and Moons/Pluto/Pluto (48).jpg  
      inflating: Planets and Moons/Pluto/Pluto (49).jpg  
      inflating: Planets and Moons/Pluto/Pluto (5).jpg  
      inflating: Planets and Moons/Pluto/Pluto (50).jpg  
      inflating: Planets and Moons/Pluto/Pluto (51).jpg  
      inflating: Planets and Moons/Pluto/Pluto (52).jpg  
      inflating: Planets and Moons/Pluto/Pluto (53).jpg  
      inflating: Planets and Moons/Pluto/Pluto (54).jpg  
      inflating: Planets and Moons/Pluto/Pluto (55).jpg  
      inflating: Planets and Moons/Pluto/Pluto (56).jpg  
      inflating: Planets and Moons/Pluto/Pluto (57).jpg  
      inflating: Planets and Moons/Pluto/Pluto (58).jpg  
      inflating: Planets and Moons/Pluto/Pluto (59).jpg  
      inflating: Planets and Moons/Pluto/Pluto (6).jpg  
      inflating: Planets and Moons/Pluto/Pluto (60).jpg  
      inflating: Planets and Moons/Pluto/Pluto (61).jpg  
      inflating: Planets and Moons/Pluto/Pluto (62).jpg  
      inflating: Planets and Moons/Pluto/Pluto (63).jpg  
      inflating: Planets and Moons/Pluto/Pluto (64).jpg  
      inflating: Planets and Moons/Pluto/Pluto (65).jpg  
      inflating: Planets and Moons/Pluto/Pluto (66).jpg  
      inflating: Planets and Moons/Pluto/Pluto (67).jpg  
      inflating: Planets and Moons/Pluto/Pluto (68).jpg  
      inflating: Planets and Moons/Pluto/Pluto (69).jpg  
      inflating: Planets and Moons/Pluto/Pluto (7).jpg  
      inflating: Planets and Moons/Pluto/Pluto (70).jpg  
      inflating: Planets and Moons/Pluto/Pluto (71).jpg  
      inflating: Planets and Moons/Pluto/Pluto (72).jpg  
      inflating: Planets and Moons/Pluto/Pluto (73).jpg  
      inflating: Planets and Moons/Pluto/Pluto (74).jpg  
      inflating: Planets and Moons/Pluto/Pluto (75).jpg  
      inflating: Planets and Moons/Pluto/Pluto (76).jpg  
      inflating: Planets and Moons/Pluto/Pluto (77).jpg  
      inflating: Planets and Moons/Pluto/Pluto (78).jpg  
      inflating: Planets and Moons/Pluto/Pluto (79).jpg  
      inflating: Planets and Moons/Pluto/Pluto (8).jpg  
      inflating: Planets and Moons/Pluto/Pluto (80).jpg  
      inflating: Planets and Moons/Pluto/Pluto (81).jpg  
      inflating: Planets and Moons/Pluto/Pluto (82).jpg  
      inflating: Planets and Moons/Pluto/Pluto (83).jpg  
      inflating: Planets and Moons/Pluto/Pluto (84).jpg  
      inflating: Planets and Moons/Pluto/Pluto (85).jpg  
      inflating: Planets and Moons/Pluto/Pluto (86).jpg  
      inflating: Planets and Moons/Pluto/Pluto (87).jpg  
      inflating: Planets and Moons/Pluto/Pluto (88).jpg  
      inflating: Planets and Moons/Pluto/Pluto (89).jpg  
      inflating: Planets and Moons/Pluto/Pluto (9).jpg  
      inflating: Planets and Moons/Pluto/Pluto (90).jpg  
      inflating: Planets and Moons/Pluto/Pluto (91).jpg  
      inflating: Planets and Moons/Pluto/Pluto (92).jpg  
      inflating: Planets and Moons/Pluto/Pluto (93).jpg  
      inflating: Planets and Moons/Pluto/Pluto (94).jpg  
      inflating: Planets and Moons/Pluto/Pluto (95).jpg  
      inflating: Planets and Moons/Pluto/Pluto (96).jpg  
      inflating: Planets and Moons/Pluto/Pluto (97).jpg  
      inflating: Planets and Moons/Pluto/Pluto (98).jpg  
      inflating: Planets and Moons/Pluto/Pluto (99).jpg  
       creating: Planets and Moons/Saturn/
      inflating: Planets and Moons/Saturn/Saturn (1).jpg  
      inflating: Planets and Moons/Saturn/Saturn (10).jpg  
      inflating: Planets and Moons/Saturn/Saturn (100).jpg  
      inflating: Planets and Moons/Saturn/Saturn (101).jpg  
      inflating: Planets and Moons/Saturn/Saturn (102).jpg  
      inflating: Planets and Moons/Saturn/Saturn (103).jpg  
      inflating: Planets and Moons/Saturn/Saturn (104).jpg  
      inflating: Planets and Moons/Saturn/Saturn (105).jpg  
      inflating: Planets and Moons/Saturn/Saturn (106).jpg  
      inflating: Planets and Moons/Saturn/Saturn (107).jpg  
      inflating: Planets and Moons/Saturn/Saturn (108).jpg  
      inflating: Planets and Moons/Saturn/Saturn (109).jpg  
      inflating: Planets and Moons/Saturn/Saturn (11).jpg  
      inflating: Planets and Moons/Saturn/Saturn (110).jpg  
      inflating: Planets and Moons/Saturn/Saturn (111).jpg  
      inflating: Planets and Moons/Saturn/Saturn (112).jpg  
      inflating: Planets and Moons/Saturn/Saturn (113).jpg  
      inflating: Planets and Moons/Saturn/Saturn (114).jpg  
      inflating: Planets and Moons/Saturn/Saturn (115).jpg  
      inflating: Planets and Moons/Saturn/Saturn (116).jpg  
      inflating: Planets and Moons/Saturn/Saturn (117).jpg  
      inflating: Planets and Moons/Saturn/Saturn (118).jpg  
      inflating: Planets and Moons/Saturn/Saturn (119).jpg  
      inflating: Planets and Moons/Saturn/Saturn (12).jpg  
      inflating: Planets and Moons/Saturn/Saturn (120).jpg  
      inflating: Planets and Moons/Saturn/Saturn (121).jpg  
      inflating: Planets and Moons/Saturn/Saturn (122).jpg  
      inflating: Planets and Moons/Saturn/Saturn (123).jpg  
      inflating: Planets and Moons/Saturn/Saturn (124).jpg  
      inflating: Planets and Moons/Saturn/Saturn (125).jpg  
      inflating: Planets and Moons/Saturn/Saturn (126).jpg  
      inflating: Planets and Moons/Saturn/Saturn (127).jpg  
      inflating: Planets and Moons/Saturn/Saturn (128).jpg  
      inflating: Planets and Moons/Saturn/Saturn (129).jpg  
      inflating: Planets and Moons/Saturn/Saturn (13).jpg  
      inflating: Planets and Moons/Saturn/Saturn (130).jpg  
      inflating: Planets and Moons/Saturn/Saturn (131).jpg  
      inflating: Planets and Moons/Saturn/Saturn (132).jpg  
      inflating: Planets and Moons/Saturn/Saturn (133).jpg  
      inflating: Planets and Moons/Saturn/Saturn (134).jpg  
      inflating: Planets and Moons/Saturn/Saturn (135).jpg  
      inflating: Planets and Moons/Saturn/Saturn (136).jpg  
      inflating: Planets and Moons/Saturn/Saturn (137).jpg  
      inflating: Planets and Moons/Saturn/Saturn (138).jpg  
      inflating: Planets and Moons/Saturn/Saturn (139).jpg  
      inflating: Planets and Moons/Saturn/Saturn (14).jpg  
      inflating: Planets and Moons/Saturn/Saturn (140).jpg  
      inflating: Planets and Moons/Saturn/Saturn (141).jpg  
      inflating: Planets and Moons/Saturn/Saturn (142).jpg  
      inflating: Planets and Moons/Saturn/Saturn (143).jpg  
      inflating: Planets and Moons/Saturn/Saturn (144).jpg  
      inflating: Planets and Moons/Saturn/Saturn (145).jpg  
      inflating: Planets and Moons/Saturn/Saturn (146).jpg  
      inflating: Planets and Moons/Saturn/Saturn (147).jpg  
      inflating: Planets and Moons/Saturn/Saturn (148).jpg  
      inflating: Planets and Moons/Saturn/Saturn (149).jpg  
      inflating: Planets and Moons/Saturn/Saturn (15).jpg  
      inflating: Planets and Moons/Saturn/Saturn (16).jpg  
      inflating: Planets and Moons/Saturn/Saturn (17).jpg  
      inflating: Planets and Moons/Saturn/Saturn (18).jpg  
      inflating: Planets and Moons/Saturn/Saturn (19).jpg  
      inflating: Planets and Moons/Saturn/Saturn (2).jpg  
      inflating: Planets and Moons/Saturn/Saturn (20).jpg  
      inflating: Planets and Moons/Saturn/Saturn (21).jpg  
      inflating: Planets and Moons/Saturn/Saturn (22).jpg  
      inflating: Planets and Moons/Saturn/Saturn (23).jpg  
      inflating: Planets and Moons/Saturn/Saturn (24).jpg  
      inflating: Planets and Moons/Saturn/Saturn (25).jpg  
      inflating: Planets and Moons/Saturn/Saturn (26).jpg  
      inflating: Planets and Moons/Saturn/Saturn (27).jpg  
      inflating: Planets and Moons/Saturn/Saturn (28).jpg  
      inflating: Planets and Moons/Saturn/Saturn (29).jpg  
      inflating: Planets and Moons/Saturn/Saturn (3).jpg  
      inflating: Planets and Moons/Saturn/Saturn (30).jpg  
      inflating: Planets and Moons/Saturn/Saturn (31).jpg  
      inflating: Planets and Moons/Saturn/Saturn (32).jpg  
      inflating: Planets and Moons/Saturn/Saturn (33).jpg  
      inflating: Planets and Moons/Saturn/Saturn (34).jpg  
      inflating: Planets and Moons/Saturn/Saturn (35).jpg  
      inflating: Planets and Moons/Saturn/Saturn (36).jpg  
      inflating: Planets and Moons/Saturn/Saturn (37).jpg  
      inflating: Planets and Moons/Saturn/Saturn (38).jpg  
      inflating: Planets and Moons/Saturn/Saturn (39).jpg  
      inflating: Planets and Moons/Saturn/Saturn (4).jpg  
      inflating: Planets and Moons/Saturn/Saturn (40).jpg  
      inflating: Planets and Moons/Saturn/Saturn (41).jpg  
      inflating: Planets and Moons/Saturn/Saturn (42).jpg  
      inflating: Planets and Moons/Saturn/Saturn (43).jpg  
      inflating: Planets and Moons/Saturn/Saturn (44).jpg  
      inflating: Planets and Moons/Saturn/Saturn (45).jpg  
      inflating: Planets and Moons/Saturn/Saturn (46).jpg  
      inflating: Planets and Moons/Saturn/Saturn (47).jpg  
      inflating: Planets and Moons/Saturn/Saturn (48).jpg  
      inflating: Planets and Moons/Saturn/Saturn (49).jpg  
      inflating: Planets and Moons/Saturn/Saturn (5).jpg  
      inflating: Planets and Moons/Saturn/Saturn (50).jpg  
      inflating: Planets and Moons/Saturn/Saturn (51).jpg  
      inflating: Planets and Moons/Saturn/Saturn (52).jpg  
      inflating: Planets and Moons/Saturn/Saturn (53).jpg  
      inflating: Planets and Moons/Saturn/Saturn (54).jpg  
      inflating: Planets and Moons/Saturn/Saturn (55).jpg  
      inflating: Planets and Moons/Saturn/Saturn (56).jpg  
      inflating: Planets and Moons/Saturn/Saturn (57).jpg  
      inflating: Planets and Moons/Saturn/Saturn (58).jpg  
      inflating: Planets and Moons/Saturn/Saturn (59).jpg  
      inflating: Planets and Moons/Saturn/Saturn (6).jpg  
      inflating: Planets and Moons/Saturn/Saturn (60).jpg  
      inflating: Planets and Moons/Saturn/Saturn (61).jpg  
      inflating: Planets and Moons/Saturn/Saturn (62).jpg  
      inflating: Planets and Moons/Saturn/Saturn (63).jpg  
      inflating: Planets and Moons/Saturn/Saturn (64).jpg  
      inflating: Planets and Moons/Saturn/Saturn (65).jpg  
      inflating: Planets and Moons/Saturn/Saturn (66).jpg  
      inflating: Planets and Moons/Saturn/Saturn (67).jpg  
      inflating: Planets and Moons/Saturn/Saturn (68).jpg  
      inflating: Planets and Moons/Saturn/Saturn (69).jpg  
      inflating: Planets and Moons/Saturn/Saturn (7).jpg  
      inflating: Planets and Moons/Saturn/Saturn (70).jpg  
      inflating: Planets and Moons/Saturn/Saturn (71).jpg  
      inflating: Planets and Moons/Saturn/Saturn (72).jpg  
      inflating: Planets and Moons/Saturn/Saturn (73).jpg  
      inflating: Planets and Moons/Saturn/Saturn (74).jpg  
      inflating: Planets and Moons/Saturn/Saturn (75).jpg  
      inflating: Planets and Moons/Saturn/Saturn (76).jpg  
      inflating: Planets and Moons/Saturn/Saturn (77).jpg  
      inflating: Planets and Moons/Saturn/Saturn (78).jpg  
      inflating: Planets and Moons/Saturn/Saturn (79).jpg  
      inflating: Planets and Moons/Saturn/Saturn (8).jpg  
      inflating: Planets and Moons/Saturn/Saturn (80).jpg  
      inflating: Planets and Moons/Saturn/Saturn (81).jpg  
      inflating: Planets and Moons/Saturn/Saturn (82).jpg  
      inflating: Planets and Moons/Saturn/Saturn (83).jpg  
      inflating: Planets and Moons/Saturn/Saturn (84).jpg  
      inflating: Planets and Moons/Saturn/Saturn (85).jpg  
      inflating: Planets and Moons/Saturn/Saturn (86).jpg  
      inflating: Planets and Moons/Saturn/Saturn (87).jpg  
      inflating: Planets and Moons/Saturn/Saturn (88).jpg  
      inflating: Planets and Moons/Saturn/Saturn (89).jpg  
      inflating: Planets and Moons/Saturn/Saturn (9).jpg  
      inflating: Planets and Moons/Saturn/Saturn (90).jpg  
      inflating: Planets and Moons/Saturn/Saturn (91).jpg  
      inflating: Planets and Moons/Saturn/Saturn (92).jpg  
      inflating: Planets and Moons/Saturn/Saturn (93).jpg  
      inflating: Planets and Moons/Saturn/Saturn (94).jpg  
      inflating: Planets and Moons/Saturn/Saturn (95).jpg  
      inflating: Planets and Moons/Saturn/Saturn (96).jpg  
      inflating: Planets and Moons/Saturn/Saturn (97).jpg  
      inflating: Planets and Moons/Saturn/Saturn (98).jpg  
      inflating: Planets and Moons/Saturn/Saturn (99).jpg  
       creating: Planets and Moons/Uranus/
      inflating: Planets and Moons/Uranus/Uranus (1).jpg  
      inflating: Planets and Moons/Uranus/Uranus (10).jpg  
      inflating: Planets and Moons/Uranus/Uranus (100).jpg  
      inflating: Planets and Moons/Uranus/Uranus (101).jpg  
      inflating: Planets and Moons/Uranus/Uranus (102).jpg  
      inflating: Planets and Moons/Uranus/Uranus (103).jpg  
      inflating: Planets and Moons/Uranus/Uranus (104).jpg  
      inflating: Planets and Moons/Uranus/Uranus (105).jpg  
      inflating: Planets and Moons/Uranus/Uranus (106).jpg  
      inflating: Planets and Moons/Uranus/Uranus (107).jpg  
      inflating: Planets and Moons/Uranus/Uranus (108).jpg  
      inflating: Planets and Moons/Uranus/Uranus (109).jpg  
      inflating: Planets and Moons/Uranus/Uranus (11).jpg  
      inflating: Planets and Moons/Uranus/Uranus (110).jpg  
      inflating: Planets and Moons/Uranus/Uranus (111).jpg  
      inflating: Planets and Moons/Uranus/Uranus (112).jpg  
      inflating: Planets and Moons/Uranus/Uranus (113).jpg  
      inflating: Planets and Moons/Uranus/Uranus (114).jpg  
      inflating: Planets and Moons/Uranus/Uranus (115).jpg  
      inflating: Planets and Moons/Uranus/Uranus (116).jpg  
      inflating: Planets and Moons/Uranus/Uranus (117).jpg  
      inflating: Planets and Moons/Uranus/Uranus (118).jpg  
      inflating: Planets and Moons/Uranus/Uranus (119).jpg  
      inflating: Planets and Moons/Uranus/Uranus (12).jpg  
      inflating: Planets and Moons/Uranus/Uranus (120).jpg  
      inflating: Planets and Moons/Uranus/Uranus (121).jpg  
      inflating: Planets and Moons/Uranus/Uranus (122).jpg  
      inflating: Planets and Moons/Uranus/Uranus (123).jpg  
      inflating: Planets and Moons/Uranus/Uranus (124).jpg  
      inflating: Planets and Moons/Uranus/Uranus (125).jpg  
      inflating: Planets and Moons/Uranus/Uranus (126).jpg  
      inflating: Planets and Moons/Uranus/Uranus (127).jpg  
      inflating: Planets and Moons/Uranus/Uranus (128).jpg  
      inflating: Planets and Moons/Uranus/Uranus (129).jpg  
      inflating: Planets and Moons/Uranus/Uranus (13).jpg  
      inflating: Planets and Moons/Uranus/Uranus (130).jpg  
      inflating: Planets and Moons/Uranus/Uranus (131).jpg  
      inflating: Planets and Moons/Uranus/Uranus (132).jpg  
      inflating: Planets and Moons/Uranus/Uranus (133).jpg  
      inflating: Planets and Moons/Uranus/Uranus (134).jpg  
      inflating: Planets and Moons/Uranus/Uranus (135).jpg  
      inflating: Planets and Moons/Uranus/Uranus (136).jpg  
      inflating: Planets and Moons/Uranus/Uranus (137).jpg  
      inflating: Planets and Moons/Uranus/Uranus (138).jpg  
      inflating: Planets and Moons/Uranus/Uranus (139).jpg  
      inflating: Planets and Moons/Uranus/Uranus (14).jpg  
      inflating: Planets and Moons/Uranus/Uranus (140).jpg  
      inflating: Planets and Moons/Uranus/Uranus (141).jpg  
      inflating: Planets and Moons/Uranus/Uranus (142).jpg  
      inflating: Planets and Moons/Uranus/Uranus (143).jpg  
      inflating: Planets and Moons/Uranus/Uranus (144).jpg  
      inflating: Planets and Moons/Uranus/Uranus (145).jpg  
      inflating: Planets and Moons/Uranus/Uranus (146).jpg  
      inflating: Planets and Moons/Uranus/Uranus (147).jpg  
      inflating: Planets and Moons/Uranus/Uranus (148).jpg  
      inflating: Planets and Moons/Uranus/Uranus (149).jpg  
      inflating: Planets and Moons/Uranus/Uranus (15).jpg  
      inflating: Planets and Moons/Uranus/Uranus (16).jpg  
      inflating: Planets and Moons/Uranus/Uranus (17).jpg  
      inflating: Planets and Moons/Uranus/Uranus (18).jpg  
      inflating: Planets and Moons/Uranus/Uranus (19).jpg  
      inflating: Planets and Moons/Uranus/Uranus (2).jpg  
      inflating: Planets and Moons/Uranus/Uranus (20).jpg  
      inflating: Planets and Moons/Uranus/Uranus (21).jpg  
      inflating: Planets and Moons/Uranus/Uranus (22).jpg  
      inflating: Planets and Moons/Uranus/Uranus (23).jpg  
      inflating: Planets and Moons/Uranus/Uranus (24).jpg  
      inflating: Planets and Moons/Uranus/Uranus (25).jpg  
      inflating: Planets and Moons/Uranus/Uranus (26).jpg  
      inflating: Planets and Moons/Uranus/Uranus (27).jpg  
      inflating: Planets and Moons/Uranus/Uranus (28).jpg  
      inflating: Planets and Moons/Uranus/Uranus (29).jpg  
      inflating: Planets and Moons/Uranus/Uranus (3).jpg  
      inflating: Planets and Moons/Uranus/Uranus (30).jpg  
      inflating: Planets and Moons/Uranus/Uranus (31).jpg  
      inflating: Planets and Moons/Uranus/Uranus (32).jpg  
      inflating: Planets and Moons/Uranus/Uranus (33).jpg  
      inflating: Planets and Moons/Uranus/Uranus (34).jpg  
      inflating: Planets and Moons/Uranus/Uranus (35).jpg  
      inflating: Planets and Moons/Uranus/Uranus (36).jpg  
      inflating: Planets and Moons/Uranus/Uranus (37).jpg  
      inflating: Planets and Moons/Uranus/Uranus (38).jpg  
      inflating: Planets and Moons/Uranus/Uranus (39).jpg  
      inflating: Planets and Moons/Uranus/Uranus (4).jpg  
      inflating: Planets and Moons/Uranus/Uranus (40).jpg  
      inflating: Planets and Moons/Uranus/Uranus (41).jpg  
      inflating: Planets and Moons/Uranus/Uranus (42).jpg  
      inflating: Planets and Moons/Uranus/Uranus (43).jpg  
      inflating: Planets and Moons/Uranus/Uranus (44).jpg  
      inflating: Planets and Moons/Uranus/Uranus (45).jpg  
      inflating: Planets and Moons/Uranus/Uranus (46).jpg  
      inflating: Planets and Moons/Uranus/Uranus (47).jpg  
      inflating: Planets and Moons/Uranus/Uranus (48).jpg  
      inflating: Planets and Moons/Uranus/Uranus (49).jpg  
      inflating: Planets and Moons/Uranus/Uranus (5).jpg  
      inflating: Planets and Moons/Uranus/Uranus (50).jpg  
      inflating: Planets and Moons/Uranus/Uranus (51).jpg  
      inflating: Planets and Moons/Uranus/Uranus (52).jpg  
      inflating: Planets and Moons/Uranus/Uranus (53).jpg  
      inflating: Planets and Moons/Uranus/Uranus (54).jpg  
      inflating: Planets and Moons/Uranus/Uranus (55).jpg  
      inflating: Planets and Moons/Uranus/Uranus (56).jpg  
      inflating: Planets and Moons/Uranus/Uranus (57).jpg  
      inflating: Planets and Moons/Uranus/Uranus (58).jpg  
      inflating: Planets and Moons/Uranus/Uranus (59).jpg  
      inflating: Planets and Moons/Uranus/Uranus (6).jpg  
      inflating: Planets and Moons/Uranus/Uranus (60).jpg  
      inflating: Planets and Moons/Uranus/Uranus (61).jpg  
      inflating: Planets and Moons/Uranus/Uranus (62).jpg  
      inflating: Planets and Moons/Uranus/Uranus (63).jpg  
      inflating: Planets and Moons/Uranus/Uranus (64).jpg  
      inflating: Planets and Moons/Uranus/Uranus (65).jpg  
      inflating: Planets and Moons/Uranus/Uranus (66).jpg  
      inflating: Planets and Moons/Uranus/Uranus (67).jpg  
      inflating: Planets and Moons/Uranus/Uranus (68).jpg  
      inflating: Planets and Moons/Uranus/Uranus (69).jpg  
      inflating: Planets and Moons/Uranus/Uranus (7).jpg  
      inflating: Planets and Moons/Uranus/Uranus (70).jpg  
      inflating: Planets and Moons/Uranus/Uranus (71).jpg  
      inflating: Planets and Moons/Uranus/Uranus (72).jpg  
      inflating: Planets and Moons/Uranus/Uranus (73).jpg  
      inflating: Planets and Moons/Uranus/Uranus (74).jpg  
      inflating: Planets and Moons/Uranus/Uranus (75).jpg  
      inflating: Planets and Moons/Uranus/Uranus (76).jpg  
      inflating: Planets and Moons/Uranus/Uranus (77).jpg  
      inflating: Planets and Moons/Uranus/Uranus (78).jpg  
      inflating: Planets and Moons/Uranus/Uranus (79).jpg  
      inflating: Planets and Moons/Uranus/Uranus (8).jpg  
      inflating: Planets and Moons/Uranus/Uranus (80).jpg  
      inflating: Planets and Moons/Uranus/Uranus (81).jpg  
      inflating: Planets and Moons/Uranus/Uranus (82).jpg  
      inflating: Planets and Moons/Uranus/Uranus (83).jpg  
      inflating: Planets and Moons/Uranus/Uranus (84).jpg  
      inflating: Planets and Moons/Uranus/Uranus (85).jpg  
      inflating: Planets and Moons/Uranus/Uranus (86).jpg  
      inflating: Planets and Moons/Uranus/Uranus (87).jpg  
      inflating: Planets and Moons/Uranus/Uranus (88).jpg  
      inflating: Planets and Moons/Uranus/Uranus (89).jpg  
      inflating: Planets and Moons/Uranus/Uranus (9).jpg  
      inflating: Planets and Moons/Uranus/Uranus (90).jpg  
      inflating: Planets and Moons/Uranus/Uranus (91).jpg  
      inflating: Planets and Moons/Uranus/Uranus (92).jpg  
      inflating: Planets and Moons/Uranus/Uranus (93).jpg  
      inflating: Planets and Moons/Uranus/Uranus (94).jpg  
      inflating: Planets and Moons/Uranus/Uranus (95).jpg  
      inflating: Planets and Moons/Uranus/Uranus (96).jpg  
      inflating: Planets and Moons/Uranus/Uranus (97).jpg  
      inflating: Planets and Moons/Uranus/Uranus (98).jpg  
      inflating: Planets and Moons/Uranus/Uranus (99).jpg  
       creating: Planets and Moons/Venus/
      inflating: Planets and Moons/Venus/Venus (1).jpg  
      inflating: Planets and Moons/Venus/Venus (10).jpg  
      inflating: Planets and Moons/Venus/Venus (100).jpg  
      inflating: Planets and Moons/Venus/Venus (101).jpg  
      inflating: Planets and Moons/Venus/Venus (102).jpg  
      inflating: Planets and Moons/Venus/Venus (103).jpg  
      inflating: Planets and Moons/Venus/Venus (104).jpg  
      inflating: Planets and Moons/Venus/Venus (105).jpg  
      inflating: Planets and Moons/Venus/Venus (106).jpg  
      inflating: Planets and Moons/Venus/Venus (107).jpg  
      inflating: Planets and Moons/Venus/Venus (108).jpg  
      inflating: Planets and Moons/Venus/Venus (109).jpg  
      inflating: Planets and Moons/Venus/Venus (11).jpg  
      inflating: Planets and Moons/Venus/Venus (110).jpg  
      inflating: Planets and Moons/Venus/Venus (111).jpg  
      inflating: Planets and Moons/Venus/Venus (112).jpg  
      inflating: Planets and Moons/Venus/Venus (113).jpg  
      inflating: Planets and Moons/Venus/Venus (114).jpg  
      inflating: Planets and Moons/Venus/Venus (115).jpg  
      inflating: Planets and Moons/Venus/Venus (116).jpg  
      inflating: Planets and Moons/Venus/Venus (117).jpg  
      inflating: Planets and Moons/Venus/Venus (118).jpg  
      inflating: Planets and Moons/Venus/Venus (119).jpg  
      inflating: Planets and Moons/Venus/Venus (12).jpg  
      inflating: Planets and Moons/Venus/Venus (120).jpg  
      inflating: Planets and Moons/Venus/Venus (121).jpg  
      inflating: Planets and Moons/Venus/Venus (122).jpg  
      inflating: Planets and Moons/Venus/Venus (123).jpg  
      inflating: Planets and Moons/Venus/Venus (124).jpg  
      inflating: Planets and Moons/Venus/Venus (125).jpg  
      inflating: Planets and Moons/Venus/Venus (126).jpg  
      inflating: Planets and Moons/Venus/Venus (127).jpg  
      inflating: Planets and Moons/Venus/Venus (128).jpg  
      inflating: Planets and Moons/Venus/Venus (129).jpg  
      inflating: Planets and Moons/Venus/Venus (13).jpg  
      inflating: Planets and Moons/Venus/Venus (130).jpg  
      inflating: Planets and Moons/Venus/Venus (131).jpg  
      inflating: Planets and Moons/Venus/Venus (132).jpg  
      inflating: Planets and Moons/Venus/Venus (133).jpg  
      inflating: Planets and Moons/Venus/Venus (134).jpg  
      inflating: Planets and Moons/Venus/Venus (135).jpg  
      inflating: Planets and Moons/Venus/Venus (136).jpg  
      inflating: Planets and Moons/Venus/Venus (137).jpg  
      inflating: Planets and Moons/Venus/Venus (138).jpg  
      inflating: Planets and Moons/Venus/Venus (139).jpg  
      inflating: Planets and Moons/Venus/Venus (14).jpg  
      inflating: Planets and Moons/Venus/Venus (140).jpg  
      inflating: Planets and Moons/Venus/Venus (141).jpg  
      inflating: Planets and Moons/Venus/Venus (142).jpg  
      inflating: Planets and Moons/Venus/Venus (143).jpg  
      inflating: Planets and Moons/Venus/Venus (144).jpg  
      inflating: Planets and Moons/Venus/Venus (145).jpg  
      inflating: Planets and Moons/Venus/Venus (146).jpg  
      inflating: Planets and Moons/Venus/Venus (147).jpg  
      inflating: Planets and Moons/Venus/Venus (148).jpg  
      inflating: Planets and Moons/Venus/Venus (149).jpg  
      inflating: Planets and Moons/Venus/Venus (15).jpg  
      inflating: Planets and Moons/Venus/Venus (16).jpg  
      inflating: Planets and Moons/Venus/Venus (17).jpg  
      inflating: Planets and Moons/Venus/Venus (18).jpg  
      inflating: Planets and Moons/Venus/Venus (19).jpg  
      inflating: Planets and Moons/Venus/Venus (2).jpg  
      inflating: Planets and Moons/Venus/Venus (20).jpg  
      inflating: Planets and Moons/Venus/Venus (21).jpg  
      inflating: Planets and Moons/Venus/Venus (22).jpg  
      inflating: Planets and Moons/Venus/Venus (23).jpg  
      inflating: Planets and Moons/Venus/Venus (24).jpg  
      inflating: Planets and Moons/Venus/Venus (25).jpg  
      inflating: Planets and Moons/Venus/Venus (26).jpg  
      inflating: Planets and Moons/Venus/Venus (27).jpg  
      inflating: Planets and Moons/Venus/Venus (28).jpg  
      inflating: Planets and Moons/Venus/Venus (29).jpg  
      inflating: Planets and Moons/Venus/Venus (3).jpg  
      inflating: Planets and Moons/Venus/Venus (30).jpg  
      inflating: Planets and Moons/Venus/Venus (31).jpg  
      inflating: Planets and Moons/Venus/Venus (32).jpg  
      inflating: Planets and Moons/Venus/Venus (33).jpg  
      inflating: Planets and Moons/Venus/Venus (34).jpg  
      inflating: Planets and Moons/Venus/Venus (35).jpg  
      inflating: Planets and Moons/Venus/Venus (36).jpg  
      inflating: Planets and Moons/Venus/Venus (37).jpg  
      inflating: Planets and Moons/Venus/Venus (38).jpg  
      inflating: Planets and Moons/Venus/Venus (39).jpg  
      inflating: Planets and Moons/Venus/Venus (4).jpg  
      inflating: Planets and Moons/Venus/Venus (40).jpg  
      inflating: Planets and Moons/Venus/Venus (41).jpg  
      inflating: Planets and Moons/Venus/Venus (42).jpg  
      inflating: Planets and Moons/Venus/Venus (43).jpg  
      inflating: Planets and Moons/Venus/Venus (44).jpg  
      inflating: Planets and Moons/Venus/Venus (45).jpg  
      inflating: Planets and Moons/Venus/Venus (46).jpg  
      inflating: Planets and Moons/Venus/Venus (47).jpg  
      inflating: Planets and Moons/Venus/Venus (48).jpg  
      inflating: Planets and Moons/Venus/Venus (49).jpg  
      inflating: Planets and Moons/Venus/Venus (5).jpg  
      inflating: Planets and Moons/Venus/Venus (50).jpg  
      inflating: Planets and Moons/Venus/Venus (51).jpg  
      inflating: Planets and Moons/Venus/Venus (52).jpg  
      inflating: Planets and Moons/Venus/Venus (53).jpg  
      inflating: Planets and Moons/Venus/Venus (54).jpg  
      inflating: Planets and Moons/Venus/Venus (55).jpg  
      inflating: Planets and Moons/Venus/Venus (56).jpg  
      inflating: Planets and Moons/Venus/Venus (57).jpg  
      inflating: Planets and Moons/Venus/Venus (58).jpg  
      inflating: Planets and Moons/Venus/Venus (59).jpg  
      inflating: Planets and Moons/Venus/Venus (6).jpg  
      inflating: Planets and Moons/Venus/Venus (60).jpg  
      inflating: Planets and Moons/Venus/Venus (61).jpg  
      inflating: Planets and Moons/Venus/Venus (62).jpg  
      inflating: Planets and Moons/Venus/Venus (63).jpg  
      inflating: Planets and Moons/Venus/Venus (64).jpg  
      inflating: Planets and Moons/Venus/Venus (65).jpg  
      inflating: Planets and Moons/Venus/Venus (66).jpg  
      inflating: Planets and Moons/Venus/Venus (67).jpg  
      inflating: Planets and Moons/Venus/Venus (68).jpg  
      inflating: Planets and Moons/Venus/Venus (69).jpg  
      inflating: Planets and Moons/Venus/Venus (7).jpg  
      inflating: Planets and Moons/Venus/Venus (70).jpg  
      inflating: Planets and Moons/Venus/Venus (71).jpg  
      inflating: Planets and Moons/Venus/Venus (72).jpg  
      inflating: Planets and Moons/Venus/Venus (73).jpg  
      inflating: Planets and Moons/Venus/Venus (74).jpg  
      inflating: Planets and Moons/Venus/Venus (75).jpg  
      inflating: Planets and Moons/Venus/Venus (76).jpg  
      inflating: Planets and Moons/Venus/Venus (77).jpg  
      inflating: Planets and Moons/Venus/Venus (78).jpg  
      inflating: Planets and Moons/Venus/Venus (79).jpg  
      inflating: Planets and Moons/Venus/Venus (8).jpg  
      inflating: Planets and Moons/Venus/Venus (80).jpg  
      inflating: Planets and Moons/Venus/Venus (81).jpg  
      inflating: Planets and Moons/Venus/Venus (82).jpg  
      inflating: Planets and Moons/Venus/Venus (83).jpg  
      inflating: Planets and Moons/Venus/Venus (84).jpg  
      inflating: Planets and Moons/Venus/Venus (85).jpg  
      inflating: Planets and Moons/Venus/Venus (86).jpg  
      inflating: Planets and Moons/Venus/Venus (87).jpg  
      inflating: Planets and Moons/Venus/Venus (88).jpg  
      inflating: Planets and Moons/Venus/Venus (89).jpg  
      inflating: Planets and Moons/Venus/Venus (9).jpg  
      inflating: Planets and Moons/Venus/Venus (90).jpg  
      inflating: Planets and Moons/Venus/Venus (91).jpg  
      inflating: Planets and Moons/Venus/Venus (92).jpg  
      inflating: Planets and Moons/Venus/Venus (93).jpg  
      inflating: Planets and Moons/Venus/Venus (94).jpg  
      inflating: Planets and Moons/Venus/Venus (95).jpg  
      inflating: Planets and Moons/Venus/Venus (96).jpg  
      inflating: Planets and Moons/Venus/Venus (97).jpg  
      inflating: Planets and Moons/Venus/Venus (98).jpg  
      inflating: Planets and Moons/Venus/Venus (99).jpg  


## 📥 > 📙 Import to Libraries


```python
import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback, EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.layers import *
from tensorflow import keras
from tensorflow.keras import Sequential
import tensorflow as tf
```

## 📋 Data Preparing


```python
train_datagen = ImageDataGenerator(
    featurewise_center=True,
    samplewise_center=False,
    featurewise_std_normalization=True,
    samplewise_std_normalization=False,
    zca_whitening=True,
    zca_epsilon=1e-06,
    rotation_range=0,
    width_shift_range=0.0,
    height_shift_range=0.0,
    brightness_range=None,
    shear_range=0.0,
    zoom_range=0.0,
    channel_shift_range=0.0,
    fill_mode='nearest',
    cval=0.0,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=1.0/255.0,
    preprocessing_function=None,
    data_format=None,
    dtype=None,
    validation_split=0.1)
train_generator = train_datagen.flow_from_directory("/content/Planets and Moons",target_size=(32, 32),
                                                    batch_size=128,
                                                    class_mode='categorical',
                                                    interpolation="lanczos",
                                                    subset="training")
test_generator = train_datagen.flow_from_directory("/content/Planets and Moons",target_size=(32, 32),
                                                    batch_size=128,
                                                    class_mode='categorical',
                                                    interpolation="lanczos",
                                                    subset="validation")
```

    Found 1484 images belonging to 11 classes.
    Found 154 images belonging to 11 classes.


## 🧱 Models Structure and Code [Function]


```python
def emir_model():
  inp = Input(shape = (32,32,3))

  x = Conv2D(32, (2,2), strides=(2,2), padding='same', activation='ReLU', use_bias=True)(inp)
  x = BatchNormalization()(x)
  x = SpatialDropout2D(0.2)(x)
  x = MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='same', data_format=None)(x)
  x = Conv2D(64, (2,2), strides=(2,2), padding='same', activation='ReLU', use_bias=True)(x)
  x = SpatialDropout2D(0.2)(x)
  x = Flatten()(x)
  x = Dense(32, activation='ReLU')(x)
  x = Dense(11, activation='softmax')(x)

  model = Model(inputs=inp, outputs= x)
  return model
```


```python
def emir_func(name_model):

    print('#####~Model => {} '.format(name_model))

    model = emir_model()
    model.summary()

    model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
    my_callbacks  =  [keras.callbacks.ModelCheckpoint("/content/model/model_{epoch}.h5")]
    
    history = model.fit(train_generator,
                        validation_data=test_generator,
                        epochs=16,
                        callbacks=my_callbacks,
                        verbose=0,
                        batch_size=128,)
    # Plotting Accuracy, val_accuracy, loss, val_loss
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    ax = ax.ravel()

    for i, met in enumerate(['accuracy', 'loss']):
        ax[i].plot(history.history[met])
        ax[i].plot(history.history['val_' + met])
        ax[i].set_title('Model {}'.format(met))
        ax[i].set_xlabel('epochs')
        ax[i].set_ylabel(met)
        ax[i].legend(['Train', 'Validation'])
    plt.show()
    
    # Predict Data Test
    pred = model.predict(test_generator)
    pred = np.argmax(pred,axis=1)
    labels = (train_generator.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    pred = [labels[k] for k in pred]
    
    print('\033[01m              Classification_report \033[0m')
    
    print('\033[01m              Results \033[0m')
    # Results
    results = model.evaluate(test_generator, verbose=0)
    print("    Test Loss:\033[31m \033[01m {:.5f} \033[30m \033[0m".format(results[0]))
    print("Test Accuracy:\033[32m \033[01m {:.2f}% \033[30m \033[0m".format(results[1] * 100))
    
    return results
```


```python
def func(pre,name_model):
    print('#####~Model => {} '.format(name_model))
    pre_model = name_model(input_shape=(32,32, 3),
                   include_top=False,
                   weights='imagenet',
                   pooling='avg')
    pre_model.trainable = False
    inputs = pre_model.input
    x = Dense(32, activation='relu')(pre_model.output)
    outputs = Dense(11, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss = 'categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
    my_callbacks  = [EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=5,
                              mode='auto')]
    
    history = model.fit(train_generator,validation_data=test_generator,epochs=16,callbacks=my_callbacks,verbose=0)
    # Plotting Accuracy, val_accuracy, loss, val_loss
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    ax = ax.ravel()

    for i, met in enumerate(['accuracy', 'loss']):
        ax[i].plot(history.history[met])
        ax[i].plot(history.history['val_' + met])
        ax[i].set_title('Model {}'.format(met))
        ax[i].set_xlabel('epochs')
        ax[i].set_ylabel(met)
        ax[i].legend(['Train', 'Validation'])
    plt.show()
    
    # Predict Data Test
    pred = model.predict(test_generator)
    pred = np.argmax(pred,axis=1)
    labels = (train_generator.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    pred = [labels[k] for k in pred]
    
    print('\033[01m              Classification_report \033[0m')
    
    print('\033[01m              Results \033[0m')
    # Results
    results = model.evaluate(test_generator, verbose=0)
    print("    Test Loss:\033[31m \033[01m {:.5f} \033[30m \033[0m".format(results[0]))
    print("Test Accuracy:\033[32m \033[01m {:.2f}% \033[30m \033[0m".format(results[1] * 100))
    
    return results
```

## 🏃‍♂️ Prep Models and My Model Benchmark Scores

### Model Emirhan


```python
model_name = "Planets_Moon_Detection_Artificial_Intelligence"
result_emirhan = emir_func(model_name)
```

    #####~Model => Planets_Moon_Detection_Artificial_Intelligence 
    Model: "model_21"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     input_22 (InputLayer)       [(None, 32, 32, 3)]       0         
                                                                     
     conv2d_52 (Conv2D)          (None, 16, 16, 32)        416       
                                                                     
     batch_normalization_32 (Bat  (None, 16, 16, 32)       128       
     chNormalization)                                                
                                                                     
     spatial_dropout2d_52 (Spati  (None, 16, 16, 32)       0         
     alDropout2D)                                                    
                                                                     
     max_pooling2d_50 (MaxPoolin  (None, 8, 8, 32)         0         
     g2D)                                                            
                                                                     
     conv2d_53 (Conv2D)          (None, 4, 4, 64)          8256      
                                                                     
     spatial_dropout2d_53 (Spati  (None, 4, 4, 64)         0         
     alDropout2D)                                                    
                                                                     
     flatten_17 (Flatten)        (None, 1024)              0         
                                                                     
     dense_40 (Dense)            (None, 32)                32800     
                                                                     
     dense_41 (Dense)            (None, 11)                363       
                                                                     
    =================================================================
    Total params: 41,963
    Trainable params: 41,899
    Non-trainable params: 64
    _________________________________________________________________



![png](Planets_and_Moons_Dataset_AI_in_Space_files/Planets_and_Moons_Dataset_AI_in_Space_17_1.png)


    2/2 [==============================] - 0s 45ms/step
    [01m              Classification_report [0m
    [01m              Results [0m
        Test Loss:[31m [01m 1.02306 [30m [0m
    Test Accuracy:[32m [01m 88.96% [30m [0m


### VGG19


```python
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
result_VGG19 = func(preprocess_input,VGG19)
```

    #####~Model => <function VGG19 at 0x7f7b167f3290> 



![png](Planets_and_Moons_Dataset_AI_in_Space_files/Planets_and_Moons_Dataset_AI_in_Space_19_1.png)


    WARNING:tensorflow:5 out of the last 9 calls to <function Model.make_predict_function.<locals>.predict_function at 0x7f7b06fb9f80> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.


    2/2 [==============================] - 2s 204ms/step
    [01m              Classification_report [0m
    [01m              Results [0m
        Test Loss:[31m [01m 0.56094 [30m [0m
    Test Accuracy:[32m [01m 87.01% [30m [0m


### VGG16


```python
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
result_VGG16 = func(preprocess_input,VGG16)
```

    #####~Model => <function VGG16 at 0x7f7b167eae60> 
    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
    58889256/58889256 [==============================] - 0s 0us/step



![png](Planets_and_Moons_Dataset_AI_in_Space_files/Planets_and_Moons_Dataset_AI_in_Space_21_1.png)


    WARNING:tensorflow:6 out of the last 11 calls to <function Model.make_predict_function.<locals>.predict_function at 0x7f7b0a6634d0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.


    2/2 [==============================] - 1s 162ms/step
    [01m              Classification_report [0m
    [01m              Results [0m
        Test Loss:[31m [01m 0.35996 [30m [0m
    Test Accuracy:[32m [01m 88.31% [30m [0m


### ResNet50


```python
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
result_ResNet50 = func(preprocess_input,ResNet50)
```

    #####~Model => <function ResNet50 at 0x7f7b167dd830> 
    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
    94765736/94765736 [==============================] - 1s 0us/step



![png](Planets_and_Moons_Dataset_AI_in_Space_files/Planets_and_Moons_Dataset_AI_in_Space_23_1.png)


    2/2 [==============================] - 2s 102ms/step
    [01m              Classification_report [0m
    [01m              Results [0m
        Test Loss:[31m [01m 1.67104 [30m [0m
    Test Accuracy:[32m [01m 41.56% [30m [0m


### ResNet101


```python
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications.resnet import preprocess_input
result_ResNet101 = func(preprocess_input,ResNet101)
```

    #####~Model => <function ResNet101 at 0x7f7b167dd8c0> 
    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet101_weights_tf_dim_ordering_tf_kernels_notop.h5
    171446536/171446536 [==============================] - 1s 0us/step



![png](Planets_and_Moons_Dataset_AI_in_Space_files/Planets_and_Moons_Dataset_AI_in_Space_25_1.png)


    2/2 [==============================] - 3s 200ms/step
    [01m              Classification_report [0m
    [01m              Results [0m
        Test Loss:[31m [01m 1.91499 [30m [0m
    Test Accuracy:[32m [01m 27.92% [30m [0m


### MobileNet


```python
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
result_MobileNet = func(preprocess_input,MobileNet)
```

    WARNING:tensorflow:`input_shape` is undefined or non-square, or `rows` is not in [128, 160, 192, 224]. Weights for input shape (224, 224) will be loaded as the default.


    #####~Model => <function MobileNet at 0x7f7b1683fd40> 
    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/mobilenet/mobilenet_1_0_224_tf_no_top.h5
    17225924/17225924 [==============================] - 0s 0us/step



![png](Planets_and_Moons_Dataset_AI_in_Space_files/Planets_and_Moons_Dataset_AI_in_Space_27_2.png)


    2/2 [==============================] - 1s 31ms/step
    [01m              Classification_report [0m
    [01m              Results [0m
        Test Loss:[31m [01m 1.43071 [30m [0m
    Test Accuracy:[32m [01m 64.29% [30m [0m


### DenseNet201


```python
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.applications.densenet import preprocess_input
result_DenseNet201 = func(preprocess_input,DenseNet201)
```

    #####~Model => <function DenseNet201 at 0x7f7b16829680> 
    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/densenet/densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5
    74836368/74836368 [==============================] - 1s 0us/step



![png](Planets_and_Moons_Dataset_AI_in_Space_files/Planets_and_Moons_Dataset_AI_in_Space_29_1.png)


    2/2 [==============================] - 5s 179ms/step
    [01m              Classification_report [0m
    [01m              Results [0m
        Test Loss:[31m [01m 0.05216 [30m [0m
    Test Accuracy:[32m [01m 100.00% [30m [0m


### EfficientNetB7


```python
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.applications.efficientnet import preprocess_input
result_Eff = func(preprocess_input,EfficientNetB7)
```

    #####~Model => <function EfficientNetB7 at 0x7f7b1682d710> 
    Downloading data from https://storage.googleapis.com/keras-applications/efficientnetb7_notop.h5
    258076736/258076736 [==============================] - 2s 0us/step



![png](Planets_and_Moons_Dataset_AI_in_Space_files/Planets_and_Moons_Dataset_AI_in_Space_31_1.png)


    2/2 [==============================] - 8s 328ms/step
    [01m              Classification_report [0m
    [01m              Results [0m
        Test Loss:[31m [01m 2.39793 [30m [0m
    Test Accuracy:[32m [01m 9.09% [30m [0m


## 📊 Finally Result of Table (DataFrame - Pandas)


```python
accuracy_result_table = pd.DataFrame({'Model':['Emirhan_Model','VGG16','VGG19','ResNet50','ResNet101','MobileNet',
                               'DenseNet201','EfficientNetB7'],
                      'Accuracy':[result_emirhan[1],result_VGG16[1], result_VGG19[1], result_ResNet50[1], result_ResNet101[1],
                                  result_MobileNet[1],result_DenseNet201[1],result_Eff[1]]})
```


```python
accuracy_result_table
```





  <div id="df-d4cd32a7-4d1a-4153-a98c-9ed3a1e217a2">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Emirhan_Model</td>
      <td>0.889610</td>
    </tr>
    <tr>
      <th>1</th>
      <td>VGG16</td>
      <td>0.883117</td>
    </tr>
    <tr>
      <th>2</th>
      <td>VGG19</td>
      <td>0.870130</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ResNet50</td>
      <td>0.415584</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ResNet101</td>
      <td>0.279221</td>
    </tr>
    <tr>
      <th>5</th>
      <td>MobileNet</td>
      <td>0.642857</td>
    </tr>
    <tr>
      <th>6</th>
      <td>DenseNet201</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>EfficientNetB7</td>
      <td>0.090909</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-d4cd32a7-4d1a-4153-a98c-9ed3a1e217a2')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-d4cd32a7-4d1a-4153-a98c-9ed3a1e217a2 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-d4cd32a7-4d1a-4153-a98c-9ed3a1e217a2');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
plt.figure(figsize=(12, 7))
plots = sns.barplot(x='Model', y='Accuracy', data=accuracy_result_table)
for bar in plots.patches:
    plots.annotate(format(bar.get_height(), '.2f'),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=15, xytext=(0, 9),
                   textcoords='offset points')

plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.xticks(rotation=20);
```


![png](Planets_and_Moons_Dataset_AI_in_Space_files/Planets_and_Moons_Dataset_AI_in_Space_35_0.png)



```python
loss_result_table = pd.DataFrame({'Model':['Emirhan_Model','VGG16','VGG19','ResNet50','ResNet101','MobileNet',
                               'DenseNet201','EfficientNetB7'],
                      'Loss':[result_emirhan[0],result_VGG16[0], result_VGG19[0], result_ResNet50[0], result_ResNet101[0],
                                  result_MobileNet[0],result_DenseNet201[0],result_Eff[0]]})
```


```python
loss_result_table
```





  <div id="df-1e85845b-6e92-48f5-ad09-1d6a8af9b899">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Emirhan_Model</td>
      <td>1.023061</td>
    </tr>
    <tr>
      <th>1</th>
      <td>VGG16</td>
      <td>0.359960</td>
    </tr>
    <tr>
      <th>2</th>
      <td>VGG19</td>
      <td>0.560944</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ResNet50</td>
      <td>1.671041</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ResNet101</td>
      <td>1.914990</td>
    </tr>
    <tr>
      <th>5</th>
      <td>MobileNet</td>
      <td>1.430708</td>
    </tr>
    <tr>
      <th>6</th>
      <td>DenseNet201</td>
      <td>0.052159</td>
    </tr>
    <tr>
      <th>7</th>
      <td>EfficientNetB7</td>
      <td>2.397933</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-1e85845b-6e92-48f5-ad09-1d6a8af9b899')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-1e85845b-6e92-48f5-ad09-1d6a8af9b899 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-1e85845b-6e92-48f5-ad09-1d6a8af9b899');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
plt.figure(figsize=(12, 7))
plots = sns.barplot(x='Model', y='Loss', data=loss_result_table)
for bar in plots.patches:
    plots.annotate(format(bar.get_height(), '.2f'),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=15, xytext=(0, 9),
                   textcoords='offset points')

plt.xlabel("Models")
plt.ylabel("Loss")
plt.xticks(rotation=20);
```


![png](Planets_and_Moons_Dataset_AI_in_Space_files/Planets_and_Moons_Dataset_AI_in_Space_38_0.png)


## My Model Performance [Benchmarks Scores] 🆚

### As a Loss Variable 📈

#### **Emirhan Model vs VGG16** `🆚`


```python
rate = result_VGG16[0]/result_emirhan[0]
print(f"The Model (Emirhan Model) I that created has a {rate} times higher performance loss score than the VGG16 model.")
```

    The Model (Emirhan Model) I that created has a 0.3518463645183309 times higher performance loss score than the VGG16 model.


#### **Emirhan Model vs result_VGG19** `🆚`


```python
rate = result_VGG19[0]/result_emirhan[0]
print(f"The Model (Emirhan Model) I that created has a {rate} times higher performance loss score than the VGG19 model.")
```

    The Model (Emirhan Model) I that created has a 0.5483000922622953 times higher performance loss score than the VGG19 model.


#### **Emirhan Model vs ResNet50** `🆚`


```python
rate = result_ResNet50[0]/result_emirhan[0]
print(f"The Model (Emirhan Model) I that created has a {rate} times higher performance loss score than the ResNet50 model.")
```

    The Model (Emirhan Model) I that created has a 1.6333748307806033 times higher performance loss score than the ResNet50 model.


#### **Emirhan Model vs ResNet101** `🆚`


```python
rate = result_ResNet101[0]/result_emirhan[0]
print(f"The Model (Emirhan Model) I that created has a {rate} times higher performance loss score than the ResNet101 model.")
```

    The Model (Emirhan Model) I that created has a 1.871824390757737 times higher performance loss score than the ResNet101 model.


#### **Emirhan Model vs MobileNet** `🆚`


```python
rate = result_MobileNet[0]/result_emirhan[0]
print(f"The Model (Emirhan Model) I that created has a {rate} times higher performance loss score than the MobileNet model.")
```

    The Model (Emirhan Model) I that created has a 1.3984584576139931 times higher performance loss score than the MobileNet model.


#### **Emirhan Model vs DenseNet201** `🆚`


```python
rate = result_DenseNet201[0]/result_emirhan[0]
print(f"The Model (Emirhan Model) I that created has a {rate} times higher performance loss score than the DenseNet201 model.")
```

    The Model (Emirhan Model) I that created has a 0.050983307725633045 times higher performance loss score than the DenseNet201 model.


#### **Emirhan Model vs EfficientNetB7** `🆚`


```python
rate = result_Eff[0]/result_emirhan[0]
print(f"The Model (Emirhan Model) I that created has a {rate} times higher performance loss score than the EfficientNetB7 model.")
```

    The Model (Emirhan Model) I that created has a 2.3438822454391453 times higher performance loss score than the EfficientNetB7 model.


## Prediction **`∼`**

**Prediction Data Preparing**


```python
prediction_datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-06,
    rotation_range=0,
    width_shift_range=0.0,
    height_shift_range=0.0,
    brightness_range=None,
    shear_range=0.0,
    zoom_range=0.0,
    channel_shift_range=0.0,
    fill_mode='nearest',
    cval=0.0,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=1.0/255.0,
    preprocessing_function=None,
    data_format=None,
    dtype=None)
!mkdir /content/planets
!mv /content/Planets-and-Moons-Dataset-AI-in-Space/Test_Earth /content/planets
prediction_generator = prediction_datagen.flow_from_directory("/content/planets",target_size=(256, 256),
                                                    batch_size=128,
                                                    class_mode='categorical',
                                                    interpolation="lanczos")
```

    Found 3 images belonging to 3 classes.


**Load to my model**


```python
from tensorflow.keras.models import load_model

model = load_model("/content/model/model_16.h5")
```

**Basic Prediction Algorithm :))**


```python
def prediction(model,data):
  prediction = model.predict(data)
  if prediction[0].max() == prediction[0][0]:
    print("Planet Prediction is 'Earth'! ")
  elif prediction[0].max() == prediction[0][1]:
    print("Planet Prediction is 'Jupiter'! ")
  elif prediction[0].max() == prediction[0][2]:
    print("Dwarf Planet Prediction is 'Makemake'! ")
  elif prediction[0].max() == prediction[0][3]:
    print("Planet Prediction is 'Mars'! ")
  elif prediction[0].max() == prediction[0][4]:
    print("Planet Prediction is 'Mercury'! ")
  elif prediction[0].max() == prediction[0][5]:
    print("Moon Prediction is 'Moon'! ")
  elif prediction[0].max() == prediction[0][6]:
    print("Planet Prediction is 'Neptune'! ")
  elif prediction[0].max() == prediction[0][7]:
    print("Dwarf Planet Prediction is 'Pluto'! ")
  elif prediction[0].max() == prediction[0][8]:
    print("Planet Prediction is 'Saturn'! ")
  elif prediction[0].max() == prediction[0][9]:
    print("Planet Prediction is 'Uranus'! ")
  elif prediction[0].max() == prediction[0][10]:
    print("Planet Prediction is 'Venus'! ")
```

***Predict to data***


```python
prediction(model,prediction_generator)
```

    1/1 [==============================] - 0s 340ms/step
    Planet Prediction is 'Earth'! 


**Earth Visualization**


```python
from PIL import Image
Image.open("/content/planets/Test_Earth/Earth/Earth.jpg")
```




![png](Planets_and_Moons_Dataset_AI_in_Space_files/Planets_and_Moons_Dataset_AI_in_Space_65_0.png)



**Prediction Data Preparing**


```python
prediction_datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-06,
    rotation_range=0,
    width_shift_range=0.0,
    height_shift_range=0.0,
    brightness_range=None,
    shear_range=0.0,
    zoom_range=0.0,
    channel_shift_range=0.0,
    fill_mode='nearest',
    cval=0.0,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=1.0/255.0,
    preprocessing_function=None,
    data_format=None,
    dtype=None)
!mkdir /content/Mars
!mv /content/Planets-and-Moons-Dataset-AI-in-Space/Test_Mars /content/Mars
prediction_generator = prediction_datagen.flow_from_directory("/content/Mars",target_size=(32, 32),
                                                    batch_size=128,
                                                    class_mode='categorical',
                                                    interpolation="lanczos")
```

    Found 1 images belonging to 1 classes.


***Predict to data***


```python
prediction(model,prediction_generator)
```

    1/1 [==============================] - 0s 109ms/step
    Planet Prediction is 'Mars'! 


**Mars Visualization**


```python
from PIL import Image
Image.open("/content/Mars/Test_Mars/Mars/Mars.jpg")
```




![png](Planets_and_Moons_Dataset_AI_in_Space_files/Planets_and_Moons_Dataset_AI_in_Space_71_0.png)



**Prediction Data Preparing**


```python
prediction_datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-06,
    rotation_range=0,
    width_shift_range=0.0,
    height_shift_range=0.0,
    brightness_range=None,
    shear_range=0.0,
    zoom_range=0.0,
    channel_shift_range=0.0,
    fill_mode='nearest',
    cval=0.0,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=1.0/255.0,
    preprocessing_function=None,
    data_format=None,
    dtype=None)
!mkdir /content/Jupiter
!mv /content/Planets-and-Moons-Dataset-AI-in-Space/Test_Jupiter /content/Jupiter
prediction_generator = prediction_datagen.flow_from_directory("/content/Jupiter",target_size=(32, 32),
                                                    batch_size=128,
                                                    class_mode='categorical',
                                                    interpolation="lanczos")
```

    Found 1 images belonging to 1 classes.


***Predict to data***


```python
prediction(model,prediction_generator)
```

    1/1 [==============================] - 0s 80ms/step
    Planet Prediction is 'Jupiter'! 


**Jupiter Visualization**


```python
from PIL import Image
Image.open("/content/Jupiter/Test_Jupiter/Jupiter/Jupiter.jpg")
```




![png](Planets_and_Moons_Dataset_AI_in_Space_files/Planets_and_Moons_Dataset_AI_in_Space_77_0.png)


