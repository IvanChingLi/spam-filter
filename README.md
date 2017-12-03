# spam-filter
I implemented two methods of making a spam filter. The first is hand-coding batch gradient descent logistic regression. The second is using TensorFlow estimator LinearRegressor. There are 3000 labeled training data samples and 2572 labeled test data samples.

For batch gradient descent logistic regression, we used the logistic loss function:

<a href="http://www.codecogs.com/eqnedit.php?latex=E(\vec{w})&space;=&space;\Sigma&space;(t_n&space;\ln(y_n)&plus;(1-t_n)\ln(1-y_n))" target="_blank"><img src="http://latex.codecogs.com/gif.latex?E(\vec{w})&space;=&space;\Sigma&space;(t_n&space;\ln(y_n)&plus;(1-t_n)\ln(1-y_n))" title="E(\vec{w}) = \Sigma (t_n \ln(y_n)+(1-t_n)\ln(1-y_n))" /></a>

where t_n is the true value (1 for spam and 0 for ham) and y_n is the predicted value from the weights (real number between 0 and 1). The predicted value is obtained using the sigmoid function:

<a href="http://www.codecogs.com/eqnedit.php?latex=\sigma(\vec{w}\cdot\vec{\phi})&space;=&space;\frac{1}{1&plus;\exp[-\vec{w}\cdot\vec{\phi}]}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\sigma(\vec{w}\cdot\vec{\phi})&space;=&space;\frac{1}{1&plus;\exp[-\vec{w}\cdot\vec{\phi}]}" title="\sigma(\vec{w}\cdot\vec{\phi}) = \frac{1}{1+\exp[-\vec{w}\cdot\vec{\phi}]}" /></a>

where phi is a vector representation of the instance.

Each step is calculated using:

<a href="http://www.codecogs.com/eqnedit.php?latex=\vec{w}_{n&plus;1}&space;=&space;\vec{w}_n&space;-&space;\eta&space;\nabla&space;E(\vec{w})" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\vec{w}_{n&plus;1}&space;=&space;\vec{w}_n&space;-&space;\eta&space;\nabla&space;E(\vec{w})" title="\vec{w}_{n+1} = \vec{w}_n - \eta \nabla E(\vec{w})" /></a>.
