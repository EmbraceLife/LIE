"""
Concept Explanation
- GPU
- sample
- batch
- epoch

"""


"""
### How should I cite Keras?

Please cite Keras in your publications if it helps your research. Here is an example BibTeX entry:

```
@misc{chollet2015keras,
  title={Keras},
  author={Chollet, Fran\c{c}ois and others},
  year={2015},
  publisher={GitHub},
  howpublished={\url{https://github.com/fchollet/keras}},
}
```
"""

"""
### How can I run Keras on GPU?

If you are running on the TensorFlow or CNTK backends, your code will automatically run on GPU if any available GPU is detected.

"""

"""
### What does "sample", "batch", "epoch" mean?

Below are some common definitions that are necessary to know and understand to correctly utilize Keras:

- **Sample**: one element of a dataset.
  - *Example:* one image is a **sample** in a convolutional network
  - *Example:* one audio file is a **sample** for a speech recognition model

- **Batch**: a set of *N* samples. The samples in a **batch** are processed independently, in parallel. If training, a batch results in only one update to the model.
  - A **batch** generally approximates the distribution of the input data better than a single input. The larger the batch, the better the approximation; however, it is also true that the batch will take longer to processes and will still result in only one update. For inference (evaluate/predict), it is recommended to pick a batch size that is as large as you can afford without going out of memory (since larger batches will usually result in faster evaluating/prediction).

- **Epoch**: an arbitrary cutoff, generally defined as "one pass over the entire dataset", used to separate training into distinct phases, which is useful for logging and periodic evaluation.
  - When using `evaluation_data` or `evaluation_split` with the `fit` method of Keras models, evaluation will be run at the end of every **epoch**.
  - Within Keras, there is the ability to add [callbacks](https://keras.io/callbacks/) specifically designed to be run at the end of an **epoch**. Examples of these are learning rate changes and model checkpointing (saving).
"""
