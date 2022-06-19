# KerasCV Bounding Boxes

All KerasCV components that process bounding boxes require a `bounding_box_format`
argument.  This argument allows you to seamlessly integrate KerasCV components into
your own workflows while preserving proper behavior of the components themselves.

The bounding box formats supported in KerasCV 
[are listed in the API docs](/api/keras_cv/bounding_box/formats)
If a format you would like to use is missing,
[feel free to open a GitHub issue on KerasCV](https://github.com/keras-team/keras-cv/issues)!
