from tensorflow.contrib.keras.python.keras.preprocessing.image import DirectoryIterator, ImageDataGenerator


###########################
# from images in folders to image_batch_iterator
###########################
data_path_train = "/Users/Natsume/Downloads/data_for_all/dogscats/sample/train"

train_batches = DirectoryIterator(directory = data_path_train,
							   image_data_generator=ImageDataGenerator(),
							   # images from folder are made into arrays with shape (224, 224), as in ImageNet images
							   target_size=(224, 224),
							   color_mode = "rgb", # as ImageNet
							#  classes=["dogs", "cats"], # optional
							   class_mode="categorical",
							   # for 2 or more classes;
							   # binary for only 2 classes
							   batch_size=32, # as you like
							   shuffle=True, # shuffle for each batch selection
							   seed=123,
							   data_format="channels_last"
							#    save_to_dir
							#    save_prefix
							#    save_format
							   )
img, lab = train_batches.next()

--------------


Init signature: tf.contrib.keras.preprocessing.image.DirectoryIterator(directory, image_da_generator, target_size=(256, 256), color_mode='rgb', classes=None, class_mode='categocal', batch_size=32, shuffle=True, seed=None, data_format=None, save_to_dir=None, save_efix='', save_format='png', follow_links=False)
Source:
class DirectoryIterator(Iterator):
  """Iterator capable of reading images from a directory on disk.

  Arguments:
      directory: Path to the directory to read images from.
          via the `classes` argument.
      image_data_generator: Instance of `ImageDataGenerator`
          to use for random transformations and normalization.
      target_size: tuple of integers, dimensions to resize input images to.
      color_mode: One of `"rgb"`, `"grayscale"`. Color mode to read images.
      classes: Optional list of strings, names of sudirectories
          containing images from each class (e.g. `["dogs", "cats"]`).
          It will be computed automatically if not set.
      class_mode: Mode for yielding the targets:
          `"binary"`: binary targets (if there are only two classes),
          `"categorical"`: categorical targets,
          `"sparse"`: integer targets,
          `"input"`: targets are images identical to input images (mainly
              used to work with autoencoders),
          `None`: no targets get yielded (only input images are yielded).
      batch_size: Integer, size of a batch.
      shuffle: Boolean, whether to shuffle the data between epochs.
      seed: Random seed for data shuffling.
      data_format: String, one of `channels_first`, `channels_last`.
      save_to_dir: Optional directory where to save the pictures
          being yielded, in a viewable format. This is useful
          for visualizing the random transformations being
          applied, for debugging purposes.
      save_prefix: String prefix to use for saving sample
          images (if `save_to_dir` is set).
      save_format: Format to use for saving sample images
          (if `save_to_dir` is set).
  """

  def __init__(self,
               directory,
			   # use ImageDataGenerator() default: 1. set channels_last, and shape (batch, row, col, channel); 2. set mean, std, zoom_range
               image_data_generator,
               target_size=(256, 256),
               color_mode='rgb',
               classes=None,
               class_mode='categorical',
               batch_size=32,
               shuffle=True,
               seed=None,
               data_format=None,
               save_to_dir=None,
               save_prefix='',
               save_format='png',
               follow_links=False):
	# set data_format: either channels_last or channels_first
    if data_format is None:
      data_format = K.image_data_format()

	 # set data_path as directory attr
    self.directory = directory

	# take outcome of ImageDataGenerator()
    self.image_data_generator = image_data_generator

	# set target image size
    self.target_size = tuple(target_size)
    if color_mode not in {'rgb', 'grayscale'}:
      raise ValueError('Invalid color mode:', color_mode,
                       '; expected "rgb" or "grayscale".')
    # set color mode: rgb or grayscale
    self.color_mode = color_mode

	# set data_format: channels_last or first
    self.data_format = data_format

	# set image_shape based on color_mode and data_format
    if self.color_mode == 'rgb':
      if self.data_format == 'channels_last':
        self.image_shape = self.target_size + (3,)
      else:
        self.image_shape = (3,) + self.target_size
    else:
      if self.data_format == 'channels_last':
        self.image_shape = self.target_size + (1,)
      else:
        self.image_shape = (1,) + self.target_size

	# set attr: classes empty list
    self.classes = classes

	# make sure class_mode is one of the 5 options
    if class_mode not in {'categorical', 'binary', 'sparse', 'input', None}:
      raise ValueError('Invalid class_mode:', class_mode,
                       '; expected one of "categorical", '
                       '"binary", "sparse", "input"'
                       ' or None.')

    # set attr: class_mode
    self.class_mode = class_mode

	# setting for saving
    self.save_to_dir = save_to_dir
    self.save_prefix = save_prefix
    self.save_format = save_format

    white_list_formats = {'png', 'jpg', 'jpeg', 'bmp'}

    # first, count the number of samples and classes
    self.samples = 0

	# set attr: num_class, and class_indices, classes
	# num_class = 2, class_indices = {'cats': 0, 'dogs': 1}, classes = ['cats', 'dogs']
    if not classes:
      classes = []
      for subdir in sorted(os.listdir(directory)):
        if os.path.isdir(os.path.join(directory, subdir)):
          classes.append(subdir)
    self.num_class = len(classes)
    self.class_indices = dict(zip(classes, range(len(classes))))

	# recursive func on path and list
    def _recursive_list(subpath):
      return sorted(
          os.walk(subpath, followlinks=follow_links), key=lambda tpl: tpl[0])

	# set attr: samples, number of samples in the all directories used
    for subdir in classes:
      subpath = os.path.join(directory, subdir)
	  # root is directory of cats or dogs
	  # files are all filenames of each directory
      for root, _, files in _recursive_list(subpath):
		# check each filename if has image extension, count 1
        for fname in files:
          is_valid = False
          for extension in white_list_formats:
            if fname.lower().endswith('.' + extension):
              is_valid = True
              break
          if is_valid:
            self.samples += 1
    print('Found %d images belonging to %d classes.' % (self.samples,
                                                        self.num_class))

    # second, build an index of the images in the different class subfolders
	# in the end, self.filenames = ['cats/cat.10046.jpg', 'cats/cat.10138.jpg', ...., 'dogs/dog.9828.jpg', 'dogs/dog.9889.jpg']
    self.filenames = []
    self.classes = np.zeros((self.samples,), dtype='int32')
    i = 0
    for subdir in classes:
      subpath = os.path.join(directory, subdir)
      for root, _, files in _recursive_list(subpath):
        for fname in files:
          is_valid = False
          for extension in white_list_formats:
            if fname.lower().endswith('.' + extension):
              is_valid = True
              break
          if is_valid:
			# give each image a class index 0 or 1
            self.classes[i] = self.class_indices[subdir]
            i += 1
            # add filename relative to directory
            absolute_path = os.path.join(root, fname)
			# store only 'cats/cat.10046.jpg' alike into self.fileneames
            self.filenames.append(os.path.relpath(absolute_path, directory))

	# initialize class Iterator(object), see image.py source code
    super(DirectoryIterator, self).__init__(self.samples, batch_size, shuffle, seed)

  def next(self):
    """For python 2.x.

    Returns:
        The next batch.
    """
    with self.lock:
	  # run _flow_index() from image.py source code, return below
	  # index_array: a shuffled batch index
	  # current index for the batch at use
	  # the batch_size of current batch
      index_array, current_index, current_batch_size = next(
          self.index_generator

    # The transformation of images is not under thread lock
    # so it can be done in parallel

	# create empty | 0s array for batch_x or batch_images
    batch_x = np.zeros(
        (current_batch_size,) + self.image_shape, dtype=K.floatx())

    grayscale = self.color_mode == 'grayscale'

    # build batch of image data
    for i, j in enumerate(index_array):
	  # get the image file name
      fname = self.filenames[j]

	  # load image from the image file
	  # <PIL.Image.Image image mode=RGB size=224x224 at 0x1196DEC50>
      img = load_img(
          os.path.join(self.directory, fname),
          grayscale=grayscale,
          target_size=self.target_size)

	  # convert image to array (224, 224, 3)
      x = img_to_array(img, data_format=self.data_format)

	  # randomly augment this image array
      x = self.image_data_generator.random_transform(x)

	  # Apply the normalization configuration to a batch of inputs
      x = self.image_data_generator.standardize(x)

      batch_x[i] = x

    # optionally save augmented images to disk for debugging purposes
    if self.save_to_dir:
      for i in range(current_batch_size):
        img = array_to_img(batch_x[i], self.data_format, scale=True)
        fname = '{prefix}_{index}_{hash}.{format}'.format(
            prefix=self.save_prefix,
            index=current_index + i,
            hash=np.random.randint(1e4),
            format=self.save_format)
        img.save(os.path.join(self.save_to_dir, fname))
    # build batch of labels
    if self.class_mode == 'input':
      batch_y = batch_x.copy()
    elif self.class_mode == 'sparse':
      batch_y = self.classes[index_array]
    elif self.class_mode == 'binary':
      batch_y = self.classes[index_array].astype(K.floatx())

	  # how to prepare a batch of batch_y or label
    elif self.class_mode == 'categorical':
      batch_y = np.zeros((len(batch_x), self.num_class), dtype=K.floatx())
      for i, label in enumerate(self.classes[index_array]):
        batch_y[i, label] = 1.

    else:
      return batch_x
    return batch_x, batch_y
File:           ~/miniconda2/envs/kur/lib/python3.6/site-packages/tensorflow/contrib/kes/python/keras/preprocessing/image.py
Type:           type
(END)

(Pdb++) sources train_batches.reset
(['  def reset(self):\n', '    self.batch_index = 0\n'], 777)

------------

class Iterator(object):
  """Abstract base class for image data iterators.

  Arguments:
      n: Integer, total number of samples in the dataset to loop over.
      batch_size: Integer, size of a batch.
      shuffle: Boolean, whether to shuffle the data between epochs.
      seed: Random seeding for data shuffling.
  """

  def __init__(self, n, batch_size, shuffle, seed):
    self.n = n
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.batch_index = 0
    self.total_batches_seen = 0
    self.lock = threading.Lock()
    self.index_generator = self._flow_index(n, batch_size, shuffle, seed)

  def reset(self):
    self.batch_index = 0

  def _flow_index(self, n, batch_size=32, shuffle=False, seed=None):
    # Ensure self.batch_index is 0.
    self.reset()

    while 1:

	  # set seed
      if seed is not None:
        np.random.seed(seed + self.total_batches_seen)

	  # set index_array: shuffle 200 index
      if self.batch_index == 0:
        index_array = np.arange(n)
        if shuffle:
          index_array = np.random.permutation(n)

	  # current_index: number of samples have been used so far
      current_index = (self.batch_index * batch_size) % n

	  # set flexible batch_size for each batch, and last batch
      if n > current_index + batch_size:
        current_batch_size = batch_size
        self.batch_index += 1
      else:
        current_batch_size = n - current_index
        self.batch_index = 0
      self.total_batches_seen += 1

	  # return: random image index of a batch, current batch index, current batch size
      yield (index_array[current_index:current_index + current_batch_size],
             current_index, current_batch_size)

  def __iter__(self):  # pylint: disable=non-iterator-returned
    # Needed if we want to do something like:
    # for x, y in data_gen.flow(...):
    return self

  def __next__(self, *args, **kwargs):
    return self.next(*args, **kwargs)
