# ImageCaptioning

Computer Vision - Course Project 

Instructor: Prof. Rob Fergus

Done by Mohith Damarapati | md4289 | New York Univeristy

Describing an image is an easy and obvious task for humans. Computers find it challenging to do the same as they require to understand the content of an image and to describe that understanding in a natural language. The problem of making machines to automatically caption an image is referred as “Image Captioning”. I present a CNN-LSTM based framework to solve this problem. This model achieves a BLEU-4 score of 20.98 points on MSCOCO dataset which trails human base line by 0.72 BLEU-4 points. Experiments show that the captions generated are mostly sensible and human understandable.

Humans can give rich descriptions of a visual scene. However, machines don’t have that ability as they lack commonsense and intelligence. Thus, “Image Captioning” is considered as a challenging research problem connecting the fields computer vision and natural language processing. Describing an image is important because it is one of the crucial components of an artificial intelligent machine and it has practical benefits like aiding visually impaired.

The central idea to solve this problem is to extract features of the input image and use these features to generate a human understandable text. CNNs provide a rich representation of the features which can be used for various vision tasks like classification, localization and detection. These features contain important information about objects present in the image. We also call the process of extraction of rich image features as learning disentangled representations. These disentangled representations are given as input to sequence models like RNN.

In this project, I used ResNet-152, which is pre- trained on the ImageNet data to capture image features. Features in the last layer of ResNet-101 after removing the fully connected layer are given as inputs to an LSTM , which is a modified version of RNN. LSTMs have an ability to capture long term dependencies which is crucial while generating natural language.
