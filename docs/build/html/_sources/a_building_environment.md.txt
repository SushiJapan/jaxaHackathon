# Building Environment

## Construct Environment (Workstation)

### Environment

 - Ubuntu 18.04
 - OpenVino 2022.3.1
 - NCS2 (VPU)

### Install OpenVino

#### Runtime
Runtimeのダウンロード。OpenVinoを動かす際のCoreなシステムである。また、NSC2を認識させるためのファイル等も含まれている。

[OpenVino](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html)

LTSのため、2022.3.1を使用。

<b>※因みに非常に重要だが、本家のTutorialではなくて、ここからダウンロードしないとNCS2のデバイスファイルがない。</b>

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/95636/177db560-5c37-21fb-f9bf-60034591d51c.png)

`Download Archives`を押すと次の画面が開く。

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/95636/381152f2-7ff9-0cdb-affd-deebac690399.png)

この内、Ubuntuで利用するため、`l_openvino_toolkit_ubuntu18_2022.3.1.9227.cf2c7da5689_x86_64.tgz`をダウンロード

``` bash
$ cd ~
$ wget l_openvino_toolkit_ubuntu18_2022.3.1.9227.cf2c7da5689_x86_64.tgz
$ tar xf l_openvino_toolkit_ubuntu18_2022.3.1.9227.cf2c7da5689_x86_64.tgz
```

解凍が終わったら設置

``` bash
$ sudo mkdir -p /opt/intel
$ sudo mv l_openvino_toolkit_ubuntu18_2022.3.1.9227.cf2c7da5689_x86_64 /opt/intel/openvino_2022.3.0
```

次に依存関係のダウンロード

```bash
$ cd /opt/intel/openvino_2022.3.0
$ sudo -E ./install_dependencies/install_openvino_dependencies.sh
```

最後にアクセスしやすいようにリンク

``` bash
cd /opt/intel
sudo ln -s openvino_2022.3.0 openvino_2022
```

起動時に設定ファイルが読み込まれるように追記して、再読み込み

``` bash
$ echo `source /opt/intel/openvino_2022/setupvars.sh` > ~/.bashrc
$ source ~/.bashrc
```

次のように表示されれば成功

```
[setupvars.sh] OpenVINO environment initialized
```

#### Development tool
次のような記述でTrochやTensorflowをインストール可能

```bash
$ pip install openvino-dev[caffe,kaldi,mxnet,pytorch,onnx,tensorflow2]==2022.3.0
```

### Install NCS2

設定ファイルをコピー

``` bash
$ sudo usermod -a -G users "$(whoami)"
$ cd /opt/intel/openvino_2022/install_dependencies/
$ sudo cp 97-myriad-usbboot.rules /etc/udev/rules.d/
```

最後に設定ファイルを`udevadm`に読み込ませてOpenVinoから認識できるようにする。

``` bash
$ sudo udevadm control --reload-rules
$ sudo udevadm trigger
$ sudo ldconfig
```

### Install GPU

``` bash
$ cd /opt/intel/openvino_2022/install_dependencies/
$ sudo -E ./install_NEO_OCL_driver.sh
```


## Run of Sample Code

### ハードウェアテスト
以下のPythonコードを実行する

``` python
from openvino.runtime import Core
ie = Core()
devices = ie.available_devices

for device in devices:
    device_name = ie.get_property(device, "FULL_DEVICE_NAME")
    print(f"{device}: {device_name}")
```

次のように認識されていればOK

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/95636/74e1fd5d-d0a9-b733-4204-787a98a6262e.png)


### サンプルコードを動かす

``` bash
$ cd /opt/intel/openvino_2022/samples/python/classification_sample_async
```

まずはRequirementsのインストール

```bash
$ pip install -r requirements.txt
```

設定ファイルのダウンロード

``` bash
$ omz_downloader --name alexnet
$ omz_converter --name alexnet
```

`-d MYRIAD`または`-d CPU`でNCS2とCPUを切り替えることができる。

``` bash
$ python classification_sample_async.py -m public/alexnet/FP16/alexnet.xml -i test.jpg -d MYRIAD
```

次のように動作したらOK
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/95636/bae34b1c-abda-74fe-934c-7cdae1761e2c.png)

##  Implementation of Custom Network
YOLO v7-tinyをNCS2で動作させたい。ただし、YOLO v7のKeras版がないので、Torch版で進めていく。
最終的にProtoBufferやONNXに変換できればMOでIR形式に変換できるのでOK。

以下を使っていく。
https://github.com/WongKinYiu/yolov7


### Install Torch
TORCHは1.12.0じゃないとエラーがでる。Cudaは11.3がよい。

``` bash
$ pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
```


### Install YOLOv7
YOLOをGitからダウンロードして、依存環境をインストール

``` bash
$ git clone https://github.com/WongKinYiu/yolov7
$ cd YOLOv7
$ pip install -r requirements.txt
```

###  Training on YOLOv7
後は良しなにyoolov7-tinyのモデル構造を変更するのと伴にデータセットを作って、学習。
（余力あれば追記）

```bash
$ python train.py --workers 8 --device 0 --batch-size 8 --data data/ship.yaml --img 800 800 --cfg cfg/training/yolov7-tiny.yaml --weights '' --name yolov7_ship --hyp data/hyp.scratch.p5.yaml
```

こんな感じになればOK。画像サイズは32の倍数でないといけない。

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/95636/394a8b4d-31f9-ccc4-1c4b-fd239ef557b1.png)

学習できていれば停止
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/95636/55d60a44-a979-cebe-65cf-14e27a8d6f22.png)


### Testing on YOLOv7

```bash
$ python detect.py --weights runs/train/yolov7_ship/weights/best.pt --conf 0.8 --img-size 800 --source custom_dataset/images/val/img_99-0.png
```

次のようなログが流れて始めれば成功

```
Namespace(weights=['runs/train/yolov7_ship/weights/best.pt'], source='custom_dataset/images/val/img_99-0.png', img_size=800, conf_thres=0.8, iou_thres=0.45, device='', view_img=False, save_txt=False, save_conf=False, nosave=False, classes=None, agnostic_nms=False, augment=False, update=False, project='runs/detect', name='exp', exist_ok=False, no_trace=False)
YOLOR ? v0.1-126-g84932d7 torch 1.12.0+cu113 CUDA:0 (NVIDIA TITAN V, 12064.375MB)
                                             CUDA:1 (NVIDIA TITAN V, 12066.875MB)

Fusing layers...
IDetect.fuse
/home/taiyaki/anaconda3/envs/torch/lib/python3.9/site-packages/torch/functional.py:478: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  ../aten/src/ATen/native/TensorShape.cpp:2894.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
Model Summary: 208 layers, 6007596 parameters, 0 gradients, 13.0 GFLOPS
 Convert model to Traced-model...
 traced_script_module saved!
 model is traced!

2 ships, Done. (6.8ms) Inference, (1.2ms) NMS
```

推定画像。精度は99%程度
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/95636/046a9c26-f639-52cc-027e-4a88b3056e81.png)

