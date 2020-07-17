# Ideas

## 2020-07-16

* Next steps:
  * Create more images (multiple TNSEs and run CNN)
  * Transfer learning
  * Inception? (fine tuning)

## 2020-07-09

- interpolation of pixels in images

Martin: hyperparameter optimization CNN
Maria: transfer learning
How to layer CNNs
Convolution, pulling, concatenating
Try different pulling sizes
Different TSNE sizes and different convolution sizes (inception layers) 

## 2020-07-03

- checking pipeline
  - talk through it

## 2020-07-02

Things to play with
- Visualisations 
  - Perplexity (or whatever for UMAP) 
  - Something else other than Umap/TSNEs?
- Hyperparameters of CNN
- Data augmentation 
- Check that it makes sense: 
  - Our scaling  
- Transfer learning
- Inception layers? 
- Try different architectures on how to combine layers

## 2020-06-xx

- transfer learning (on muffins :))
  - train autoencoders on t-SNE images across datasets
  - [which transfer learning to use](https://towardsdatascience.com/transfer-learning-from-pre-trained-models-f2393f124751)

- inception layers
- multiple perplexities (data augmentation or multiple layers with connections to inception layers)
- multi omics
- CNNs as feature extractors for preprocessing (and then use RFs or whatever)

- data augmentation
  - butterfly (two head network)
  - [article on how to handle small datasets](https://towardsdatascience.com/breaking-the-curse-of-small-data-sets-in-machine-learning-part-2-894aa45277f4)

- multiple input network with differently structured inputs (e.g., numeric, categorical, image ... in our case omics with different sizes)

- single cell
  - multiple layers with mean features
  - image interpolation for cell assignments to empty pixels

- methods
  - [progressive resizing of networks](https://towardsdatascience.com/boost-your-cnn-image-classifier-performance-with-progressive-resizing-in-keras-a7d96da06e20)

