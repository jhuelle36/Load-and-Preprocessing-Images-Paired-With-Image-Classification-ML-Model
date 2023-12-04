

import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
#import tensorflow_datasets as tfds - The last few sets of instructions downloads the data using this library which uses the "resources" library which is only available to Unix-like os(Linux)


print(tf.__version__)


import pathlib

#Download images
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
archive = tf.keras.utils.get_file(origin=dataset_url, extract=True)
data_dir = pathlib.Path(archive).with_suffix('')

#Prints count of images
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

#Opens image of roses
roses = list(data_dir.glob('roses/*'))
PIL.Image.open(str(roses[0]))
roses = list(data_dir.glob('roses/*'))
PIL.Image.open(str(roses[1]))


#Batch size is the number of data points processed by the model, powers of 2's, 32 common for limited computational ressources, higher batch sizes are more computational efficient
batch_size = 32
#Size of the images, make sure they are all on same scale

img_height = 180 
img_width = 180


#Creates a training tf Dataset from image files form our directory - Holds the 80 percent of data fro training
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir, #Directory containing images
  validation_split=0.2, #20 percent of data goes towards validation, could change to .15 or .25
  subset="training", #Determines whether or not to store validation or training data within train_ds, since we want to train we will use "training"
  seed=123, #Seed allows for unique split of validation and training data that can be reused, could have used any arbitrary integer: 190
  image_size=(img_height, img_width), #Defined image sizes, could alter this if I knew dimesnions of images
  batch_size=batch_size) #Defined batch size: 32, could have used 64

#Creates a validation tf Dataset from image files form our directory - holds the 20 percent of data for validation
val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2, #This 20 percent of 
  subset="validation", #Subset tells function to store data for validation
  seed=123, #Using the same seed allows for the EXACT same split, meaning this data will be the 20 percent that was not used for training
  image_size=(img_height, img_width),
  batch_size=batch_size)



#Print out the classnames of the different types of flowers
class_names = train_ds.class_names
print(class_names)


import matplotlib.pyplot as plt

#Plot out the data
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

    
plt.show()

#Scalar -> vector -> matix -> tensor(n dimesnions)
#The batch is in the form of a tensor(multi dimesnional array): (32, 180, 180, 3) or 32 images of size 180x180x3(WidthxHeightxRGB channel)
for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

#The RGB channel is in the range of 0 to 255 so we are normalizing them to be between 0 and 1 to make it easier for the model
normalization_layer = tf.keras.layers.Rescaling(1./255) #If you changed to Rescaling(1./127.5, offset = -1), it will be a scale of -1 to 1



#Two ways to implement our normalization model, apply it to the dataset using Dataset.map or you can add it directly into your model as a layer(easier)

#Way 1
#normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y)) #Takes training data and normalizes the x values but keeps the y values the same
#image_batch, labels_batch = next(iter(normalized_ds)) #Extracts a batch from the dataset: image_batch contains normalized input, labels_batch contains corresponding labels
#Print out
#first_image = image_batch[0]
#print(np.min(first_image), np.max(first_image)) #Print min and max pixel from the firstimage of the batch


#AUTOTUNE automatically adjust buffer size depending on available system resources(Computers computational resources come runtime)
AUTOTUNE = tf.data.AUTOTUNE


#Caching increases the time to recieve data significantly. We are keeping it in an area of memory that is a lot faster to pull from - Common to cache a dataset after expensivve preproccesing data to prevent recomputing for each epoch
#buffer_size determines number of batches to prefetch
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


#Both methods decreases training times and prevent data from bottlenecking in our model:
#Buffered Prefetching - Loads a batch of data into a buffer while the model is still training on the previous batch - Reduces training time, prevents model from waiting on next batch
#Pararllel Loading - Involves loading and preprocessing multiple batches of data at the same time - Faster data loading, Makes model more efficient
  #Loading - Take data fromthe original source(file) and put into memory, and then put in the ideal format, in this case a 3D array(Tensor[l x w x RGB]).
  #Preprocessing - Normalizing data, text tokenization... etc.




num_classes = 5

model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu'), #32 filters, 3x3 kernel size, relu activation function
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(), #Converts 2D spatial information from convolutional and pooling layers to a 2D vector, this output will then be fed into dense layer
  tf.keras.layers.Dense(128, activation='relu'), #This intermediate dense layer brings it all together to find more sophisticated patterns fromr the more simpler ones in previous layers
  tf.keras.layers.Dense(num_classes) #Important for classification(prediction), fully connected, set to number of classifications(types of flowers)
])

#Hierarchy of Features:
#Here we go Dr.Reyes, I am about to cook:
# Convolutional layers - Use small filters(square matrix of weights[kernel]) to scan over local data and then slides across the entire input data at different local positions.
  #It then continues doing this and finds local patterns. This is very useful in grid-like data(cough* cough* images) for finding local patterns to eventually find larger
  #and larger patterns. That is why it is useful to have multiple layers of convolutional layers because each layer will use the patterns from the previous convolutional 
  #layers, allowing for the model to find more and more complex patterns. The inner convolutional layers will find very local shapes and edges, the middle convolutional layer
  #will find patterns from those previous shapes and edges such as textures, parts of an object, or structures and then our final layer will find patterns within those previous
  #structures such as entire objects, scence or complex textures. In our example it would go from edges and lines, to shapes of petals/stems, to the actual flower itself.

# Pooling layers - The goal of these layers are to reduce the spatial dimensions of the input data. The layer is trying to tune out the nonimportant information and hold onto
  #the more important information. It does this by looking locally and finding the important information from that region. There are two main types: average and max pooling.
  #Average pooling will hold onto the average of the region, this is helpful for finding the general presence of that region. Max pooling is finding the max of that reigion and
  #this is helpful where detecting the presence of a feature is more important than location. MaxPooling2D allows us to focus on the most important information of our input data
  #and calling it multiple times finds more complex relationships.



#Compile our model witht he Adam optimizer
model.compile(
  optimizer='adam',
  #This loss function is common for classifcation tasks when target labels are integrs and it applys the softmax from logits and calculate probalabilities
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),#from_logits = True means that the data is in logits form
  metrics=['accuracy']) #Will display accuracy when training

#Sparse data - small fraction of features is relevant to the task, a lot of zero values.


#Begin training with validation data and training data and loop through that data 3 times
model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=3
)

#Creates a dataset of file paths
list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=False)
list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)


#prints first 5 
for f in list_ds.take(5):
  print(f.numpy())


#Creates alphabetically sorted np array of files/directories except LICENSE.txt
class_names = np.array(sorted([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"]))
print(class_names)


#Split into training and validation sets
val_size = int(image_count * 0.2)
train_ds = list_ds.skip(val_size)
val_ds = list_ds.take(val_size)

#Print length
print(tf.data.experimental.cardinality(train_ds).numpy())
print(tf.data.experimental.cardinality(val_ds).numpy())


#Function that converts a file path to an (img, label) pair
def get_label(file_path):
  # Convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  one_hot = parts[-2] == class_names
  # Integer encode the label
  return tf.argmax(one_hot)


def decode_img(img):
  # Convert the compressed string to a 3D uint8 tensor
  img = tf.io.decode_jpeg(img, channels=3)
  # Resize the image to the desired size
  return tf.image.resize(img, [img_height, img_width])


def process_path(file_path):
  label = get_label(file_path)
  # Load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label

#Applys a given function to each elemeent of the dataset
train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)

for image, label in train_ds.take(1):
  print("Image shape: ", image.numpy().shape)
  print("Label: ", label.numpy())

#Applys optimizations to the input
def configure_for_performance(ds):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=1000)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds

train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)



image_batch, label_batch = next(iter(train_ds))

plt.figure(figsize=(10, 10))
for i in range(9):
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(image_batch[i].numpy().astype("uint8"))
  label = label_batch[i]
  plt.title(class_names[label])
  plt.axis("off")

  plt.show()

#Train the model
model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=3
)
