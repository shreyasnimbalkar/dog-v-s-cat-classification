# Dog vs cat classification

A small image classifier that tells dogs from cats. Instead of training a CNN from
scratch, it puts one dense layer on top of a pretrained MobileNetV2 and trains only
that layer. Everything is in one Colab notebook, `dog_v_s_cat.ipynb`.

## Data

The Kaggle [Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats) competition data.
The training archive has 25,000 labeled images. To keep it quick on Colab I used
2,000 of them (990 cats, 1,010 dogs), resized to 224x224 RGB, and split them
80/20 into 1,600 training and 400 test images.

The label comes from the file name (`dog.123.jpg` is a dog, `cat.45.jpg` is a cat).

## Approach

- MobileNetV2 feature extractor from TensorFlow Hub
  (`tf2-preview/mobilenet_v2/feature_vector/4`), frozen. It turns each image into a
  1,280-number feature vector.
- One `Dense(2)` layer on top, trained with Adam and sparse categorical cross-entropy
  on logits.
- Pixel values scaled to 0-1. No data augmentation.
- 5 epochs.

Only 2,562 of the 2,260,546 parameters are trained, which is why 1,600 images are
enough.

## Results

| | Accuracy |
|---|---|
| Training, epoch 5 | 99.4% |
| Test (400 images) | 98.0% (loss 0.057) |

The last cells load a single image from a path and print "Dog" or "Cat". I tried it
on one dog and one cat photo from the web and both came out right.

## Running it

The easiest way is the "Open in Colab" badge at the top of the notebook. Locally:

```bash
pip install tensorflow tensorflow-hub opencv-python pillow scikit-learn numpy matplotlib kaggle
```

You need a Kaggle API token (`kaggle.json`) to download the data. The notebook copies
it to `~/.kaggle/` and runs `kaggle competitions download -c dogs-vs-cats`. A few cells
use Colab-only helpers (`google.colab.patches.cv2_imshow`), so swap those for
`matplotlib` if you run it outside Colab.

## Files

```
dog_v_s_cat.ipynb   download, preprocessing, training, evaluation, prediction
LICENSE             MIT
```

## License

MIT, see [LICENSE](LICENSE).
