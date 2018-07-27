# Visualization of a CNN for character recognition with Layerwise Relevance Propagation

Try it: [stefanleupold.github.io/visual-cnn-lrp](https://stefanleupold.github.io/visual-cnn-lrp/) (works best in Chrome/Safari)

## Description
This repository contains a pretrained convolutional neural network in client-side JavaScript that detects characters 0 to 9. After prediction a layerwise relevance propagation heatmap is computed and displayed next to the input image. 

![alt text](https://github.com/StefanLeupold/visual-cnn-lrp/raw/master/Screenshot.jpg "")

## Credits
[Original code of the neural network](http://scs.ryerson.ca/~aharley/vis/) was published by A. W. Harley in "An Interactive Node-Link Visualization of Convolutional Neural Networks," in ISVC, pages 867-877, 2015.

The layerwise relevance propagation method was introduced in S Bach, A Binder, G Montavon, F Klauschen, KR Müller, W Samek: [On Pixel-wise Explanations for Non-Linear Classifier Decisions by Layer-wise Relevance Propagation](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140).

This work was supported by BMBF ALICE II grant 01IB15001B.
