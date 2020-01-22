# EMNIST-analysis
Hand written character recognition

data gotten in courtesy of 
Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). 
EMNIST: an extension of MNIST to handwritten letters. Retrieved from http://arxiv.org/abs/1702.05373 
I used the byclass data set with 62 classes. Many methods were tried that did not work better or
took enormous time to train like feature engineering: add three new variables; sum of pixels,
area of space that can hold liquid when the upright and transposed. But the analysis took too long
to train. I tried other methods rnn, included and autoencoder.  
The CNN method worked best with an accuracy of 0.769 
