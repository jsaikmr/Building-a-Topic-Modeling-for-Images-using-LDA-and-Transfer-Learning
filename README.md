# Building-a-Topic-Modeling-for-Images-using-LDA-and-Transfer-Learning
### Abstract:

Topic modeling is a technique used to extract the hidden topics from a large volume of text. There are several algorithms used for topic modeling such as Latent Dirichlet Allocation(LDA), Latent Semantic Analysis(LSA), Non-Negative Matrix Factorization(NMF), etc. However, the challenge is to extract the topics from images. This involves both the text and the image processing to extract good quality of topics. Most of the blogs have focused on detecting topics from textual information. For a change, I wanted to extend and explore topic modeling for images. This article explains the steps involved in combining both of these processing techniques to uncover the themes from images.

### Introduction:

Automatic topic modeling for images puts forth a particular challenge in computer vision and natural language processing because it needs to interpret from both the visual and the textual information which is two completely different information forms.  

In this article, we will see how we can make use of an image caption dataset to build a topic detection model for images. We will use the 'Flickr8k' dataset to keep it simple and easy to train.

We will use the Latent Dirichlet Allocation(LDA) to extract the topics from the vocabulary of caption data and pre-trained VGGNet16 model to extract the patterns from the images and then train the model to predict the topics for the given images.

Let's dive-in!

### Import Packages:

The core packages used in this article are Gensim, NLTK, Spacy, and Keras. Apart from this we also use Pandas, Numpy, Sklearn, and Matplotlib for data handling and visualization.



### Data Loading:

You can download the 'Flickr8k' dataset from here https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/.  

After extracting zip files from the downloaded dataset, you will find the below folders.

	- Flickr8k_Dataset: It contains a total of 8092 images in JPEG format with different shapes and sizes. Of which 6000 are used for training, 1000 for validation and 1000 for the test dataset.
	- Flickr8k_text: Contains text files describing train_set ,test_set and dev_set. Flickr8k.token.txt contains 5 captions for each image i.e. total 40460 captions.



Now you load the image ids and captions from the Flickr8k.token.txt file and prepare a dataset. Then group the captions by image ids to form it as a single sentence for each image.



Let's take a sneak peek at the dataset.



### Data Cleaning:

Data pre-processing and cleaning is an important part of the whole model building process. It involves the below steps.

	- Tokenize the captions using the split string function
	- Normalize the case of all tokens to lowercase
	- Remove all punctuation from tokens
	- Remove all tokens that contain numerical data
	- Remove all stopwords using NLTK corpus package



Below defines the clean_text() function that will clean the loaded caption data.



### Data Lemmatization:

Now, we lemmatize the words to its root form using Spacy and filter the words that contain only specific pos-tags like NOUN, ADJ, VERB, and ADV. This will improve the accuracy of the topic detection process.



### Understanding Latent Dirichlet Allocation(LDA):

The LDA algorithm performs more than just text summarization, it also discovers recurring topics in a document collection.

The LDA algorithm extracts a set of keywords from each text document in the collection. Documents are then clustered together to learn the recurring keywords in groups of documents. These sets of recurring keywords are then considered a topic common to several documents in the collection.

### Creating a Dictionary and Corpus:

The two main inputs to the LDA topic model are the dictionary and the corpus. We can use tools in Gensim to prepare the dictionary and the corpus for the caption data.

The first step is to load the caption data in the training dataset to create the dictionary containing a mapping of word identifiers.

Next, we iterate through the caption data to prepare the corpus containing a Term-Document frequency table.



### Finding an optimal number of topics for LDA:

To build an LDA model, we would require to find the optimal number of topics to be extracted from the caption dataset. We can use the coherence score of the LDA model to identify the optimal number of topics. 

We can iterate through the list of several topics and build the LDA model for each number of topics using Gensim's LDAMulticore class. Then load the model object to the CoherenceModel class to obtain the coherence score. The LDA model and its corresponding Coherence score can be saved to find the optimal number of topics later in the course. Finally, we can plot the results of all topics and their coherence scores for better understanding.




Once we obtain the optimal model, we can print the topics summary with the top 10 words that contribute most to each topic.



Now we check the perplexity and the coherence score of the optimal model. An ideal LDA model should have low perplexity and high coherence scores.



### Predicting Topics for Captions data:

Now that we have found the optimal LDA model, we can predict the topics for each caption data in the dataset. First, we load the corpus of training, validation and test dataset to the LDA model and obtain the resultant dataset with dominant topics, percentage of contribution and top 10 keywords of a topic for each image in the dataset.



So far, we have predicted the topics for each image using the caption data available in the dataset. Next, we will see how we can process the images and train the deep learning model with the predicted topics.

### Transfer Learning:

Transfer learning is a machine learning method where a model developed for a task is reused as the starting point for a model on a second task.

It is a popular approach in deep learning where pre-trained models are used as the starting point on computer vision and natural language processing tasks given the vast compute and time resources required to develop neural network models on these problems and from the huge jumps in the skill that they provide on related problems.

### Model Building:

We will use the pre-trained VGGNet16 model for image processing that is trained on the Imagenet dataset provided in Keras. Imagenet is a standard dataset used for classification. It contains more than 14 million images in the dataset, with little more than 21 thousand groups or classes. 

We can modify the VGGNet16 model to fit our needs. We can remove the softmax layer and attach the below layers.

	- Dense layer with 2056 units and 'tanh' activation
	- Dropout layers with 0.5 percentage
	- Dense layer with 1024 units with 'tanh' activation
	- Dropout layers with 0.5 percentage
	- Softmax layer with units of an optimal number of topics 



The choice of inclusion of layers, units and activation functions are subject to domain experience or intuition developed through experience. You can experiment with various combinations to obtain a better model. 





### Image Preprocessing:

We can use the below custom generator class to load the images and the topics and return the samples as a single batch. We use the Keras preprocessing module's load_img function to load the image with the target size of (224, 224, 3). Then convert the loaded image pixels to Numpy array format using img_to_array function. Then using Keras Vgg16's preprocess_input function, process and prepare the image to load it to the pre-trained VGGNet16 model.




### Model Training:

Now let's train the model with the training and the validation dataset for 20 epochs with a batch size of 50. Finally, we plot the results of the loss and accuracy of the model at each epoch.








### Detecting Topics for Test Image Dataset:

Next, we load the weights of the best VGGNet16 model that is trained in the above step. Then load the test dataset to the predict_generator function of the VGGNet16 model to detect the probabilities of topics for each image. Now we consider the topics with the highest probability and produce a resultant data frame with image ids and their corresponding topics detected. 



### Model Evaluation:

Let's evaluate the model by loading the true topics and the predicted topics to log_loss (cross-entropy loss), accuracy_score and Confusion_matrix function of the Sklearn metrics package.



The model with low log loss value and high accuracy score will provide better prediction results. From the above results, we can see that we have achieved an accuracy of 53.2%.

### Future scope:

	* 
Tuning the hyperparameters of the model will help produce better results.
	* 
The model can be trained with Flickr30k or MS-COCO dataset for better results.
	* 
Try out the other topic detection algorithms like LSA, NMF, etc. and compare the results.
	* 
Use the pre-trained models like VGGNet19, Google's Inception, Microsoft's ResNet, etc. to achieve better accuracy



### Summary:

In this article, we discovered the topic modeling for images and how we can use the image caption dataset to build the topic detection model. Specifically, we built the topic model using Gensim's LDA. Then we saw how to find the optimal number of topics using coherence scores and choose the optimal LDA model. Then we customized the pre-trained VGGNet16 model and trained the model to detect the topics for the given images. Finally, we saw how to generate the results and evaluate the model performance using metrics in Sklearn.

That brings us to the end of this article.

You can download the complete ipython notebook from here.

### References:

	1. https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
	2. https://machinelearningmastery.com/transfer-learning-for-deep-learning/
	3. https://machinelearningmastery.com/prepare-photo-caption-dataset-training-deep-learning-model/
	4. https://nlpforhackers.io/topic-modeling/
	5. https://arxiv.org/pdf/1807.03514.pdf
