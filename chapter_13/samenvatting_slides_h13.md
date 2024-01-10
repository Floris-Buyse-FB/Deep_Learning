# H13: Loading and Preprocessing Data with TensorFlow

## 13.1 The Data API

`Create dataeset from tensors`

```Python
X = tf.range(10)
dataset = tf.data.Dataset.from_tensor_slices(X)
```

`Chaining transformations`

```Python
dataset = dataset.repeat(3).batch(7).prefetch(1)
```

Where:

- `repeat(3)` repeats the dataset 3 times
- `batch(7)` groups the instances into batches of 7 | drop_remainder=True to drop the last batch if it is not full
- `prefetch(1)` specifies that the next batch will be prepared in the background while the current batch is being processed

`Caution:`

- These methods do not modify the dataset, they create a new one

`Map, filter, take`

```Python
dataset = dataset.map(lambda x: x * 2)
dataset = dataset.filter(lambda x: tf.reduce_sum(x) > 50)
dataset = dataset.take(3)
```

Where:

- `map()` transforms each element using the given function
- `filter()` filters out some elements based on the given predicate
- `tf.reduce_sum()` computes the sum of all the elements in the tensor
- `take()` creates a new dataset containing at most the given number of elements

### 13.1.2 Shuffling the Data

- Gradient Descent works best when the instances in the training set are independent and identically distributed (IID)

`Shuffle the dataset`

- Creates a new dataset that will start by filling up a buffer with the first items of the source dataset
- Whenever it is asked for an item, it pulls one out randomly from the buffer and replaces it with a fresh one from the source dataset, until it has iterated through the entire source dataset

```Python
dataset = tf.data.Dataset.range(10).repeat(3)
dataset = dataset.shuffle(buffer_size=5, seed=42).batch(7)
```

Where:

- `buffer_size` specifies the size of the buffer -> must be large enough or shuffling will not be very effective
- `seed` specifies the random seed
- if `repeat()` comes after `shuffle()`, then the shuffling is done after each repetition
- if "reshuffle_each_iteration=False" then order will be same after each iteration

### 13.1.3 Interleaving Lines from Multiple Files

#### Create dataset from multiple (CSV) files

- `tf.data.Dataset.list_files(train_filepaths, seed=42, shuffle=True)`
  - train_filepaths: list of filepaths
  - seed: random seed
  - shuffle: shuffle the filepaths or not (default=True)

#### With interleaving

```Python
filepath_dataset = tf.data.Dataset.list_files(train_filepaths, seed=42)
n_readers = 5
dataset = filepath_dataset.interleave(
    lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
    cycle_length=n_readers,
    num_parallel_calls=n_readers
)
```

Where:

- `skip(1)` skips the first line (header row)
- `TextLineDataset()` creates a dataset whose items are the lines of one or more text files
- `Interleave()`: reads from 5 files at a time (default)
  - cycle_length: number of files to read in parallel
  - num_parallel_calls: number of threads to use for reading records in parallel

### 13.1.4 Preprocessing the Data

Normalization without the use of build-in functions and byte strings to tensors

```Python
X_mean, X_std = np.mean(X_train, axis=0), np.std(X_train, axis=0)

def preprocess(line):
    # line: byte string conversion
    defs = [0.] * n_inputs + [tf.constant([], dtype=tf.float32)]
    fields = tf.io.decode_csv(line, record_defaults=defs)
    x = tf.stack(fields[:-1])
    y = tf.stack(fields[-1:])
    # return: normalization
    return (x - X_mean) / X_std, y

# everything together
def csv_reader_dataset(filepaths, repeat=1, n_readers=5, n_read_threads=None,
                       shuffle_buffer_size=10000, n_parse_threads=5, batch_size=32):
    dataset = tf.data.Dataset.list_files(filepaths).repeat(repeat)
    dataset = dataset.interleave(
        lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
        cycle_length=n_readers, num_parallel_calls=n_read_threads)
    dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(preprocess, num_parallel_calls=n_parse_threads)
    dataset = dataset.batch(batch_size)
    return dataset.prefetch(1)
```

`Caching`

- `cache()` method caches the dataset's items in memory (only do this if dataset fits in memory)
- Typically call `cache()` after loading and preprocessing the data, but before shuffling and batching
  - this way data will only be loaded and preprocessed once rather than once per epoch

## 13.3 Keras Preprocessing Layers

`Options:`

- Preprocessing outside of the model (Numpy, Pandas, Scikit-Learn, etc.)
- Preprocessing when loading the data (tf.data)
- Preprocessing inside the model (Keras preprocessing layers)

### 13.3.1 Normalization Layer

```Python
normalization = tf.keras.layers.Normalization()

model = tf.keras.Sequential([
    normalization,
    keras.layers.Dense(30),
    keras.layers.Dense(1)
])

# adapt() method
normalization.adapt(X_train)

model.compile(loss="mse", optimizer="nadam")
```

Where:

- `Normalization()` compute the mean and standard deviation of each feature per batch
- Notice, adapt() called before compiling the model
- But normalization layer will slow down training (data is normalized once per epoch)
  - Normalize before training -> Create in-production model with normalization layer

`Interaction with tf.data.Dataset`

```Python
normalization = tf.keras.layers.Normalization()
normalization.adapt(X_train)

dataset = dataset.map(lambda x, y: (normalization(x), y))
```

Note: here you should call cache() after map() if dataset fits in memory

### 13.3.2 Discretization Layer

- Transform numerical features into categorical features
- Provide `bin_boundaries` or let adapt() method find them
- No adapt() needed if you provide bin_boundaries
- `num_bins` specifies the number of bins to use
- Note: bin_boundaries -> left inclusive, right exclusive (here: 18 yields 1, 50 yields 2)

```Python
age = tf.constant([[11.], [22.], [33.], [92.], [18.], [50.]])
discretization = tf.keras.layers.Discretization(bin_boundaries=[18., 50.])

age_categories = discretization(age)

# yields [[0.], [1.], [1.], [2.], [1.], [2.]]
# with adapt()

discretization2 = tf.keras.layers.Discretization(num_bins=3)
discretization2.adapt(age)
```

### 13.3.3 CategoryEncoding Layer

- `tf.keras.layers.CategoryEncoding()`
- `num_tokens` specifies the number of categories

### 13.3.4 StringLookup

- To one-hot encode categorical String features
- Known categories mapped to integers (1 (most frequent) to num_categories (least frequent))

- `tf.keras.layers.StringLookup()`
- `adapt()` method necessary
- `num_oov_indices` specifies the number of out-of-vocabulary buckets (default: 1 -> all oov = 0)
- `output_mode` specifies the output type (default: "int")
  - "int": integer indices
  - "one_hot": one-hot encoding

### 13.3.5 Hashing Layer

- `tf.keras.layers.Hashing()`
- No adapt() method
- `num_bins` specifies the number of bins

### 13.3.6 Embedding Layer

- Dense representation of categorical features
- Typically initialized randomly and learned during training
- Categories with similar meaning -> similar embedding = `Representation Learning`

- `tf.keras.layers.Embedding()`
- `input_dim` specifies the number of categories
- `output_dim` specifies the dimensionality of the embedding

`Embed categorical text`

1. Create a StringLookup layer
2. Create an Embedding layer
   1. input_dim = string_lookup_layer.vocabulary_size()

### 13.3.7 Text Preprocessing

#### 13.3.7.1 TextVectorization Layer

- `tf.keras.layers.TextVectorization()`
- Pass vocabulary or let adapt() method find it
- Sentences to lower_case, remove punctuation, split on whitespaces
- Resulting words sorted by frequency
  - Most common word = 2
  - Out-of-vocabulary words = 1
  - Padding = 0

#### 13.3.7.2 TF-IDF Vectorization

- `tf.keras.layers.TextVectorization(output_mode="tf-idf")`
- `tf-idf`: term frequency-inverse document frequency
- Gives score to each word in each sentence or document
- The more frequent a word is in a sentence, the higher its score
- The more frequent a word is in the corpus, the lower its score

`Computation:`

- We have 4 documents -> d=4
- Weight for each word: idf = log(1 + d / (f + 1))
  - f = number of documents containing the word
- If say a words appears 2x in a document and in 3 documents -> tf = 2, f = 3
- tf-idf = tf \* idf => 2 \* log(1 + 4 / (3 + 1)) = 2 \* log(2) = 2 \* 0.301 = 0.602

### 13.3.8 Using Pretrained Language Models Components

- Tensorflow Hub library

### 13.3.9 Image Preprocessing

- `tf.keras.layers.Rescaling()`
- `tf.keras.layers.Resizing()`
- `tf.keras.layers.CenterCrop()`
