**Team Name**

OxMLSyS & Samsung Cambridge

**Contact email id**

edgar.liberis -at- cs.ox.ac.uk

**Model description**

We use an EfficientNet family of models [1], whose depth and width (number of channels in `Conv` outputs) are defined according to a formula (see the manuscript). We substitute lower values than those explored in the paper to obtain a model that has:

 * < 250KB peak memory usage (specifically, 229'376 B, assuming fused Conv-BN-ReLU blocks)
 * < 250KB parameters (215'040 parameters at 8 bits per parameter)
 * < 60 MB multiply-adds (specifically, 11'100'528).

The model accepts a 128x128 RGB input and its weights are quantised to 8-bit precision, using a customised version of `tf.contrib.quantize`. 
Please see [https://github.com/oxmlsys/vvw-efficientnet](https://github.com/oxmlsys/vvw-efficientnet) for more information.

**Performance metrics**

 * Accuracy on COCO minval: 83.35% (TFLite model)
 * Model size: 215'040 B
 * Peak memory usage: 229'376 B
 * Number of MAdds: 11'110'528
