# TF Face Detector

Face Detector using [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/object_detection)

![](https://user-images.githubusercontent.com/80381/29495837-2c0b05de-8602-11e7-8d38-c792e72e51d5.jpg)

Web DEMO: https://tf-face-detector.herokuapp.com/


## Prerequisite

- Python 2 (Cloud MLはpython3をサポートしていないため)
  - TensorFlow >= 1.2
  - Pillow >= 4.2.1 (for visualizing results)
  - cv2 >= 3.3 (for generating dataset)

## Setup

```
virtualenv --system-site-packages ~/tf-face-detector
source ~/tf-face-detector/bin/activate
git submodule update --init
pip install -r requirements.txt
```

## FDDB dataset

http://vis-www.cs.umass.edu/fddb/

To download data and generate tfrecord dataset (needed `cv2`):

```
python data/fddb.py
```


## Training

```sh
perl -pe "s|PATH_TO_BE_CONFIGURED|${PWD}/data|g" ./ssd_inception_v2_fddb.config.base > ssd_inception_v2_fddb.config
(cd models && protoc object_detection/protos/*.proto --python_out=.)
export PYTHONPATH=$PYTHONPATH:`pwd`/models:`pwd`/models/slim
python models/object_detection/train.py \
    --train_dir=./train \
    --pipeline_config_path=./ssd_inception_v2_fddb.config
```

or

```sh
./run_local.sh
```

## Export graph

```sh
export PYTHONPATH=${PYTHONPATH}:$(pwd)/models:$(pwd)/models/slim
export CHECKPOINT_NUMBER=<target checkpoint number>
export EXPORT_DIRECTORY=<path to output graph>
python models/object_detection/export_inference_graph.py \
    --input_type=encoded_image_string_tensor \
    --pipeline_config_path=ssd_inception_v2_fddb.config \
    --trained_checkpoint_prefix=train/model.ckpt-${CHECKPOINT_NUMBER} \
    --output_directory=${EXPORT_DIRECTORY}
```

# Google Machine Learning Engineでトレーニングする

## 注意事項

- GPUが有効なリージョンを指定してください(参考: [GPU が有効なマシンをリクエストする](https://cloud.google.com/ml-engine/docs/how-tos/using-gpus?hl=ja#requesting_gpu-enabled_machines))
- 最新の[tensorflow/models](https://github.com/tensorflow/models)だと不具合が発生するので、使わないでください。


## 1. Cloud Platform Console プロジェクトを作成してセットアップする

- 参考: [コマンドラインを使用したクイックスタート](https://cloud.google.com/ml-engine/docs/command-line?hl=ja)

上記を参考に、プロジェクトを作成して必要なセットアップをして下さい。
プロジェクト名は「tf-face-detector」にします。

## 2. パッケージング

- 参考: [Packaging](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_cloud.md#packaging)

パッケージングします

```zsh
cd models
protoc object_detection/protos/*.proto --python_out=.
python setup.py sdist
(cd slim && python setup.py sdist)
```

## 3. モデルトレーニングやバッチ予測の実行中にデータを読み書きするための Google Cloud Storage バケットを作成する

- 参考: [Cloud Storage バケットを設定する](https://cloud.google.com/ml-engine/docs/getting-started-training-prediction?hl=ja#set_up_your_cloud_storage_bucket)

```zsh
PROJECT_ID=$(gcloud config list project --format "value(core.project)")
BUCKET_NAME=${PROJECT_ID}-mlengine2
REGION=asia-east1
# バケットを作成
gsutil mb -l $REGION gs://$BUCKET_NAME
```

## 4. データファイルを Cloud Storage バケットにアップロードする

gsutil を使用して data ディレクトリを Cloud Storage バケットにコピーします。

```zsh
# TODO: data/fddbはコピーする必要がないので、除外する
gsutil cp -r data gs://$BUCKET_NAME/data
```

## 5. configファイルを Cloud Storage バケットにアップロードする

```zsh
perl -pe "s|PATH_TO_BE_CONFIGURED|gs://$BUCKET_NAME/data|g" ./ssd_inception_v2_fddb.config.base > ssd_inception_v2_fddb.config
gsutil cp -r ssd_inception_v2_fddb.config gs://$BUCKET_NAME/ssd_inception_v2_fddb.config
```

## 6. クラウドで単一インスタンスのトレーニングを実行する

- 参考: [クラウド内で単一インスタンスのトレーナーを実行する](https://cloud.google.com/ml-engine/docs/getting-started-training-prediction?hl=ja#cloud-train-single)

```zsh
JOB_NAME=object_detection_`date +%Y%m%dT%I%M%S`
TRAIN_DIR=${BUCKET_NAME}/train
OUTPUT_PATH=gs://$BUCKET_NAME/$JOB_NAME
HPTUNING_CONFIG=./hptuning_config.yaml
gcloud ml-engine jobs submit training $JOB_NAME \
    --runtime-version 1.2 \
    --job-dir $OUTPUT_PATH \
    --packages models/dist/object_detection-0.1.tar.gz,models/slim/dist/slim-0.1.tar.gz \
    --module-name object_detection.train \
    --region $REGION \
    -- \
    --train_dir=${OUTPUT_PATH}/train \
    --pipeline_config_path=gs://${BUCKET_NAME}/ssd_inception_v2_fddb.config \
    --train-steps 1000 \
    --verbosity DEBUG
```

## 7. クラウドで分散トレーニングを実行する

- 参考1: [Running a Multiworker Training Job](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_cloud.md#running-a-multiworker-training-job)
- 参考2: [クラウドで分散トレーニングを実行する](https://cloud.google.com/ml-engine/docs/getting-started-training-prediction?hl=ja#cloud-train-dist)

```zsh
JOB_NAME=object_detection_`date +%Y%m%dT%I%M%S`
TRAIN_DIR=${BUCKET_NAME}/train
OUTPUT_PATH=gs://$BUCKET_NAME/$JOB_NAME
HPTUNING_CONFIG=./hptuning_config.yaml
gcloud ml-engine jobs submit training $JOB_NAME \
    --runtime-version 1.2 \
    --job-dir $OUTPUT_PATH \
    --packages models/dist/object_detection-0.1.tar.gz,models/slim/dist/slim-0.1.tar.gz \
    --module-name object_detection.train \
    --region $REGION \
    --config $HPTUNING_CONFIG \
    -- \
    --train_dir=${OUTPUT_PATH}/train \
    --pipeline_config_path=gs://${BUCKET_NAME}/ssd_inception_v2_fddb.config
```

tensorboardを起動

```zsh
tensorboard --logdir=$OUTPUT_PATH
```

## 8. Export graph

```zsh
export CHECKPOINT_NUMBER=<target checkpoint number>
gsutil cp ${OUTPUT_PATH}/train/model.ckpt-${CHECKPOINT_NUMBER}.data-00000-of-00001 train/
gsutil cp ${OUTPUT_PATH}/train/model.ckpt-${CHECKPOINT_NUMBER}.index train/
gsutil cp ${OUTPUT_PATH}/train/model.ckpt-${CHECKPOINT_NUMBER}.meta train/

export PYTHONPATH=${PYTHONPATH}:$(pwd)/models:$(pwd)/models/slim
export EXPORT_DIRECTORY=/tmp
rm -rf $EXPORT_DIRECTORY/saved_model
python models/object_detection/export_inference_graph.py \
    --input_type=encoded_image_string_tensor \
    --pipeline_config_path=ssd_inception_v2_fddb.config \
    --trained_checkpoint_prefix=train/model.ckpt-${CHECKPOINT_NUMBER} \
    --output_directory=${EXPORT_DIRECTORY}
```
