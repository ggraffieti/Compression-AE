# Compression of MNIST data with AutoEncoders

## Guide
1. First train the CNN on MNIST to have a baseline `python train_cnn.py`
2. Then train the autoencoder to reconstruct MNIST data with `python train_simple_AE.py` (default latent space dimension = 5, you can easily change it in the training script)
3. Generate the compressed MNIST data with `python encode_data.py`
4. Train the CNN on the reconstructed compressed data with `train_cnn_encoded_data.py`

## Baselines
Using a latent space of dimension 5:


|                 | MNIST  | Reconstructed MNIST |
|-----------------|--------|---------------------|
| Accuracy        | 98.7%  | 95.1%               |
| Occupied Memory | 47.5MB | ~1.6MB\*             |

\*Compressed data = 1.2MB + Decoder weights = 434KB 

*Note* MNIST images are memorized as unsigned byte, so they use 1 byte per value. Compressed data is memorized as float, so every value use 4 byte. 
28×28×60,000 + 60,000 (labels) + some additional metadata = ~47.5MB
5×60,000×4 + 60,000 + decoder weights = ~1.6MB
Compression factor = ~30 (97% less memory required). 

## Requirements
- python >= 3.6
- pytorch >= 1.4
- torchvisio >= 0.8
