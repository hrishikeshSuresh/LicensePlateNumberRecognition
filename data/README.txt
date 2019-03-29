===============================================================================================
=                                ReId DATASET (license plates)                                =
===============================================================================================
===============================================================================================
= Paper: Holistic Recognition of Low Quality License Plates by CNN using Track Annotated Data =
= Authors: Jakub Spanhel, Jakub Sochor, Roman Juranek et al.                                  =
= E-mail: {ispanhel,isochor,ijuranek,herout}@fit.vutbr.cz                                     =
= © 2017, BUT FIT, Czech Republic                                                             =
=                                                                                             =
=                                                                                             =
= LICENSE                                                                                     =
= -------                                                                                     =
= Except where otherwise noted, this work is licensed under                                   =
= https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode                                 =
= © 2017, ŠPAŇHEL, SOCHOR, JURÁNEK, HEROUT, MARŠÍK, ZEMČÍK. Some Rights Reserved.             =
=                                                                                             =
= TERMS OF USAGE                                                                              =
= --------------                                                                              =
= Dataset is available for academic research and non-commercial use only due to license       =
= conditions. Please cite our original paper when used.                                       =
= For commercial use please contact DPGM BUT FIT at {ispanhel, herout}@fit.vutbr.cz           =
===============================================================================================


CITATION
========
@INPROCEEDINGS{Spanhel2017holistic, 
  author={{\v{S}}pa{\v{n}}hel, Jakub and Sochor, Jakub and Jur{\'a}nek, Roman and Herout, Adam and Mar{\v{s}}{\'\i}k, Luk{\'a}{\v{s}} and Zem{\v{c}}{\'\i}k, Pavel},
  booktitle={2017 14th IEEE International Conference on Advanced Video and Signal Based Surveillance (AVSS)}, 
  title={Holistic recognition of low quality license plates by CNN using track annotated data}, 
  year={2017}, 
  pages={1-6}, 
  keywords={Character recognition;Image recognition;Image segmentation;Licenses;Neural networks;Optical character recognition software;Training}, 
  organization={IEEE},
  doi={10.1109/AVSS.2017.8078501}, 
  ISBN={978-1-5386-2939-0}, 
  month={Aug},
  year={2017}
}

****************
* DATASET INFO *
****************

Ground truth labels, train/test split
============
File: trainVal.csv
------------
track_id - ID of specific track based on tracker
image_path - path to image in archive structure
lp - ground truth text for license plate
train - Train/test split. 0 - test, 1 - train
