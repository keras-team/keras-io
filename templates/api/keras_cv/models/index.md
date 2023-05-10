# KerasCV Models

KerasCV contains end-to-end implementations of popular model
architectures. These models can be created in two ways:

- Through the `from_preset()` constructor, which instantiates an object with
  a pre-trained configuration, and (optionally) weights.
  Available preset names are listed on this page.

```python
model = keras_cv.models.RetinaNet.from_preset(
    "resnet50_v2_imagenet",
    num_classes=20,
    bounding_box_format="xywh",
)
```

- Through custom configuration controlled by the user. To do this, simply
  pass the desired configuration parameters to the default constructors of the
  symbols documented below.

```python
backbone = keras_cv.models.ResNetBackbone(
    stackwise_filters=[64, 128, 256, 512],
    stackwise_blocks=[2, 2, 2, 2],
    stackwise_strides=[1, 2, 2, 2],
    include_rescaling=False,
)
model = keras_cv.models.RetinaNet(
    backbone=backbone,
    num_classes=20,
    bounding_box_format="xywh",
)
```

## Backbone presets

Each of the following preset name corresponds to a configuration and weights for
a **backbone** model.

The names below can be used with the `from_preset()` constructor for the
corresponding **backbone** model.

```python
backbone = keras_cv.models.ResNetBackbone.from_preset("resnet50_imagenet")
```

For brevity, we do not include the presets without pretrained weights in the
following table.

**Note**: All pretrained weights should be used with unnormalized pixel
intensities in the range `[0, 255]` if `include_rescaling=True` or in the range
`[0, 1]` if `including_rescaling=False`.

{{backbone_presets_table}}

## Task presets


Each of the following preset name corresponds to a configuration and weights for
a **task** model. These models are application-ready, but can be further
fine-tuned if desired.

The names below can be used with the `from_preset()` constructor for the
corresponding **task** models.

```python
object_detector = keras_cv.models.RetinaNet.from_preset(
    "retinanet_resnet50_pascalvoc",
    bounding_box_format="xywh",
)
```

Note that all backbone presets are also applicable to the tasks. For example,
you can directly use a `ResNetBackbone` preset with the `RetinaNet`. In this
case, fine-tuning is necessary since task-specific layers will be randomly
initialized.

```python
backbone = keras_cv.models.RetinaNet.from_preset(
    "resnet50_imagenet",
    bounding_box_format="xywh",
)
```

For brevity, we do not include the backbone presets in the following table.

**Note**: All pretrained weights should be used with unnormalized pixel
intensities in the range `[0, 255]` if `include_rescaling=True` or in the range
`[0, 1]` if `including_rescaling=False`.

{{task_presets_table}}

## API Documentation

{{toc}}
