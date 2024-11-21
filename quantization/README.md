## Model Quantization

First, you should install the optimum-quanto from source (we motify it):

```bash
cd optimum-quanto
pip install -e .
```

Second, you should give the path of prompts and images for calibration. You should modify the `promptpath_list` and `datapath_list` in `src/train_function.py`.

Third, you can run the following command to do quantization.

```bash
python inference_w4a8_qat.py
```
