# stlpips_metric_tflite
TFlite version of Shift-tolerant Perceptual Similarity Metric  for tensorflow

## Example:
```python
from stlpips_TF import im2tensor
from stlpips_TF import stlpips_metric

image1 = im2tensor(path=path1)
image2 = im2tensor(path=path2)


stlpips_metric_router = stlpips_metric()
stlpips_distance = stlpips_metric_router(image1, image2)

```
