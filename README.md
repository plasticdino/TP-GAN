# TP-GAN

Official TP-GAN Tensorflow implementation for the ICCV17 paper "[Beyond Face Rotation: Global and Local Perception GAN for Photorealistic and Identity Preserving Frontal View Synthesis](http://openaccess.thecvf.com/content_ICCV_2017/papers/Huang_Beyond_Face_Rotation_ICCV_2017_paper.pdf)" by [Huang, Rui](http://www.andrew.cmu.edu/user/ruih2/) and Zhang, Shu and Li, Tianyu and [He, Ran](http://www.nlpr.ia.ac.cn/english/irds/People/rhe.html).

The goal is to **recover a frontal face image of the same person from a single face image under any poses**.

###Params setting
`config\params.json`

###Dataset stuctures
|----dataset
        |----train
        |       |----train_img
        |               |----001_01_01_010_00_crop_128.png
        |               |----001_01_01_010_01_crop_128.png
        |               |---- …
        |       |----train.csv
        |       |----keypoint
        |               |----001_01_01_010.5pt
        |               |----001_01_01_041.5pt
        |               |---- …
        |----test
                |----test_img
                        |----001_01_01_010_00_crop_128.png
                        |----001_01_01_051_00_crop_128.png
                        |---- …
                |----test.csv
                |----keypoint
                        |----001_01_01_010.5pt
                        |----001_01_01_051.5pt
                        |---- …

###Train 
`train.py`
Using the option `--checkpoint` to load checkpoint (default = True)

###Test
`my_test.py`

