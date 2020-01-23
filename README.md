# EMNIST-analysis
Hand written character recognition

Data gotten in courtesy of the work done by
Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). 
EMNIST: an extension of MNIST to handwritten letters. Retrieved from http://arxiv.org/abs/1702.05373 
I used the byclass data set with 62 classes. Many methods were tried that did not work better or
took enormous time to train like feature engineering: add three new variables; sum of pixels,
area of space that can hold liquid when the upright and transposed. But the analysis took too long
to train. I tried other methods rnn, included and autoencoder.  
The CNN method worked best with an accuracy of 0.769 
This accuracy is great given that with the human eye we expect misidentification among [1, i, I,l], [2,z,Z],
[5,s,S], [0,o,O], [c,C], [k,K], [m,M], [u,U], [v.V], [w,W], [x,X] and [y,Y].
If the model guesses with these then we should expect a good accuracy to be close to
(62-[3+2+2+2+1+1+1+1+1+1+1+1])/62=0.726.
Assuming the characters are evenly distributed. We could see from the sample that m,t and f are misclassified (arguably).
From the confusion matrix we can see that there are more numbers than letters. 
With this, accuracy is still a very good result. 
