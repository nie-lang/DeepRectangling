# Deep Rectangling for Image stitching: A Learning Baseline ([paper](https://arxiv.org/abs/2203.03831))
<p align="center">Lang Nie<sup>1</sup>, Chunyu Lin<sup>1 *</sup>, Kang Liao<sup>1</sup>, Shuaicheng Liu<sup>2</sup>, Yao Zhao<sup>1</sup></p>
<p align="center"><sup>1</sup>Institute of Information Science, Beijing Jiaotong University</p>
<p align="center"><sup>2</sup>School of Information and Communication Engineering, University of Electronic Science and Technology of China</p>
<p align="center"><sup>{nielang, cylin, kang_liao, yzhao}@bjtu.edu.cn, liushuaicheng@uestc.edu.cn</sup></p>

<div align=center>
<img src="https://github.com/nie-lang/DeepRectangling/blob/main/rectangling.jpg"/>
</div>

## Dataset (DIR-D)
The details of the dataset can be found in our paper. 

We release our testing results with the proposed dataset together. One can download it at [Google Drive](https://drive.google.com/file/d/1KR5DtekPJin3bmQPlTGP4wbM1zFR80ak/view?usp=sharing) or [Baidu Cloud](https://pan.baidu.com/s/1aNpHwT8JIAfX_0GtsxsWyQ)(Extraction code: 1234).

## Requirement
* python 3.6
* numpy 1.18.1
* tensorflow 1.13.1

If you encounter some problems about the tensorflow environment, please refer to this [issue](https://github.com/nie-lang/DeepRectangling/issues/4).


## For windows system
For windows OS users, you have to change '/' to '\\\\' in 'line 52 of Codes/utils.py'.

## Training
#### Step 1: Download the pretrained vgg19 model
Download [VGG-19](https://www.vlfeat.org/matconvnet/pretrained/#downloading-the-pre-trained-models). Search imagenet-vgg-verydeep-19 in this page and download imagenet-vgg-verydeep-19.mat. 

#### Step 2: Train the network
Modify the 'Codes/constant.py' to set the 'TRAIN_FOLDER'/'ITERATIONS'/'GPU'. In our experiment, we set 'ITERATIONS' to 100,000.

```
cd Codes/
python train.py
```

## Testing
#### Pretrained model for deep rectangling
Our pretrained rectangling model can be available at [Google Drive](https://drive.google.com/drive/folders/1gEsE-7QBPcbH-kfHqYYR67C-va7vztxO?usp=sharing) or [Baidu Cloud](https://pan.baidu.com/s/19jRzz_1E97X35j6qmWm_kg)(Extraction code: 1234). And place the four files to 'Codes/checkpoints/Ptrained_model/' folder.
#### Testing 
Modidy the 'Codes/constant.py'to set the 'TEST_FOLDER'/'GPU'. The path for the checkpoint file can be modified in 'Codes/inference.py'.

```
cd Codes/
python inference.py
```
#### Testing with arbitrary resolution images
Modidy the 'Codes_for_Arbitrary_Resolution/constant.py'to set the 'TEST_FOLDER'/'GPU'. The path for the checkpoint file can be modified in 'Codes_for_Arbitrary_Resolution/inference.py'. 
Then, put the testing images into the folder 'Codes_for_Arbitrary_Resolution/other_dataset/' (including input and mask) and run:

```
cd Codes_for_Arbitrary_Resolution/
python inference.py
```
The rectangling results can be found in Codes_for_Arbitrary_Resolution/rectangling/.


## Citation
This paper has been accepted by CVPR2022 as oral presentation. If you have any questions, please feel free to contact me.


NIE Lang -- nielang@bjtu.edu.cn
```
@inproceedings{nie2022deep,
  title={Deep Rectangling for Image Stitching: A Learning Baseline},
  author={Nie, Lang and Lin, Chunyu and Liao, Kang and Liu, Shuaicheng and Zhao, Yao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5740--5748},
  year={2022}
}
```

## Reference
1. Lang Nie, Chunyu Lin, Kang Liao, Shuaicheng Liu, and Yao Zhao. Depth-aware multi-grid deep homography estimation with contextual correlation. IEEE Trans. on Circuits and Systems for Video Technology, 2021.

2. Kaiming He, Huiwen Chang, Jian Sun. Rectangling panoramic images via warping. SIGGRAPH, 2013.
