# KerasCV Bounding Boxes

All KerasCV components that process bounding boxes require a `bounding_box_format`
argument.  This argument allows you to seamlessly integrate KerasCV components into
your own workflows while preserving proper behavior of the components themselves.

Bounding boxes are represented by dictionaries with two keys: `'boxes'` and `'classes'`:

```
{
  'boxes': [batch, num_boxes, 4],
  'classes': [batch, num_boxes]
}
```

To ensure your bounding boxes comply with the KerasCV specification, you can use [`keras_cv.bounding_box.validate_format(boxes)`](https://github.com/keras-team/keras-cv/blob/master/keras_cv/bounding_box/validate_format.py).

The bounding box formats supported in KerasCV
[are listed in the API docs](/api/keras_cv/bounding_box/formats)
If a format you would like to use is missing,
[feel free to open a GitHub issue on KerasCV](https://github.com/keras-team/keras-cv/issues)!
