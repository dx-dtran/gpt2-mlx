# gpt2-mlx

Run and train GPT-2 on Apple silicon

## Quickstart

### Install

```shell
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

### Run

Download the pre-trained GPT-2 model weights from [Hugging Face](https://huggingface.co/gpt2-xl)

Convert the PyTorch model weights to MLX
```shell
$ python convert_weights.py --weights_path="path/to/pytorch_model.bin" --model_name="gpt2-xl"
```

Run the model
```shell
$ python generate.py --model_name="gpt2-xl" --prompt "In a shocking finding, scientists discovered a herd of unicorns"
```

### Train
```shell
$ python train.py
```
