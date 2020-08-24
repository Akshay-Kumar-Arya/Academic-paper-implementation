# Summary of Deep Residual Learning for image recognition


Deeper neural networks are difficult to train and Deep residual learning ease the training with help of Deep residual architecture. 
This paper deals with the degradation problem in which accuracy gets saturated while training deep neural networks. 
The intuition behind the solution is that for a certain depth the accuracy is increasing and after that it starts to decay because it is harder to form 
identity mapping with a stack of non-linear layers. Therefore, they introduced identity mapping through shortcut connections between the stacked layers. 
This mapping forces the solvers to learn residual functions `(f(x) = H(x) - x)`. If Identity mapping is the optimal formulation, the learned weights 
should be derived to zero to make f(x) equal to zero(they observed that this is a suitable preconditioning as most residual function responses are small).

Shortcut connections introduced in the networks don’t have additional parameters. The building block of identity mapping is `H(x) =  F(Wi, x) + x` 
(where F(x) = W2 * activation(W1 * x) for two layers skipped) when input/output dimensions are the same. If we change input/output dimensions the mapping 
becomes `H(x) =  F(Wi, x) + Ws * x` (where F(x) = W2 * activation(W1 * x) for two layers skipped). Similarly one can apply them for skipping multiple layers. 
The experiments showed that identity mapping is itself sufficient for addressing the degradation problem and is economical. Use Ws only when there is a need 
to change dimensions.

## Dimension Handling
Residual architecture includes identity shortcuts which can be directly used when input/output dimensions are the same. When dimensions increase (i.e. number of 
channels), two options are considered (A) The shortcut still performs identity mapping, with extra zero entries padded for increasing dimensions. This option 
introduces no extra parameter; (B) The projection shortcut (i.e. use of Ws) is used to match dimensions (done by 1×1 convolutions). For both options, when the 
shortcuts go across feature maps of two sizes, they are performed with a stride of 2.

## Bottleneck Architecture
To reduce the computational complexity, Deeper Bottleneck Architecture is introduced. For each residual function F, they use a stack of 3 layers instead of 2. 
The three layers are 1×1, 3×3, and 1×1 convolutions, where the 1×1 layers are responsible for reducing and then increasing (restoring) dimensions, leaving 
the 3×3 layer a bottleneck with smaller input/output dimensions.

## Results
ResNets are more deeper and more accurate yet computationally cheaper than other models like VGG etc. ResNets with layers upto 1200 can be easily trained. 
A single ResNet outperforms previous state-of-the-art ensembles. ResNets are not much effective when the depth of the model is small but when depth is increased, 
they perform outstanding with a significant difference as compared to planar networks. Residual learning eases optimization and helps in faster training of deep 
neural networks. It also decreases the error rate of deep networks.
