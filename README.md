# WAYLA:-what-are-you-looking-at
# Introduction
The goal of this project is to predict what image features (i.e. objects) a normal person would look at on a given image, as indicated by the project name. 

Nowadays, eye tracking technique is commonly used in visual neuroscience and cognitive science to answer visual attention related questions  since some studies have shown that human gaze behavior is able to reflect cognitive processes of the mind, such as intention, and is influenced by the user’s task. Therefore, when an image containing many different image features is provided, it is useful to know how attention of a normal person is biased towards the image features for some studies related to cognitive process. This motivates us to work on this project, which is predicting maps for images indicating attention bias of people (i.e. saliency map) by using neural network.

Another motivation is that we believe it is interesting to show that there can be artificial intelligence (AI) similar to human intelligence in terms of having attention bias when inspecting images. The term attention bias means people would spend different amount of attention/time on objects in an image. For example, when a picture of kids on playground is shown, people would pay more attention to the kids rather than the playground. If the same image is provided to the AI and it also "focuses" more on the kids, then we can conclude they have similar attention bias on this image. By doing so, we can show that it is possible for AI to be as smart as human brains in this area.\\

# User Instruction
To generate image feature predictions, there are a couple steps:
1. Generate eye gaze fixation predictions by running the main.py by python 3.6.
2. Download pre-trained Mask_RCNN model at https://github.com/matterport/Mask_RCNN/tree/master/mrcnn, then replace the /mrcnn/visualize.py by the visualize.py file in this project.
3. Run the classifier.py.
