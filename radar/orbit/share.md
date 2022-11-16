## Random Errors for Pointing

Some notes on my understanding of random errors for pointing in relation to the pointing engineering handbook.

At any point in time, a random pointing error $\mathcal{E}(t)$, corresponding to a particular axis of ratation, could be defined as a stochastic process. As a stochastic process, at time instant, $t$, $\mathcal{E}(t)$ would represent a random variable with some probability distribution function or, more generally, probability measure. 

### Payload parameters

Oftentimes, the definition of a probability distrubution for $\mathcal{E}(t)$ depends upon a set of parameters on the payload/satellite. Let these parameters be grouped into the vector $\vec{\theta}$. To be specific, these parameters could include entries, such as

$$
\vec{\theta}
=
\begin{bmatrix}
\text{x-coordinate of momentum wheel 1}\\
\text{y-coordinate of momentum wheel 1}\\
\text{z-coordinate of momentum wheel 1}\\
\text{x-coordinate of momentum wheel 2}\\
\text{y-coordinate of momentum wheel 2}\\
\text{z-coordinate of momentum wheel 2}\\
\text{x-coordinate of the center of mass}\\
\text{y-coordinate of the center of mass}\\
\text{z-coordinate of the center of mass}\\
\vdots
\end{bmatrix}
$$ 

One could then, theoretically, define the probability distribution $\text{d}F_{\mathcal{E}(t)|\vec\theta}\left(e(t)|\vec\theta\right)$. The mean value of $\mathcal{E}(t)$ may, for instance, be a function of these parameters. 

### Random parameters and a compound distribution

For any actually manufactured (realized) satellite, $\vec{\theta}$ will assume certain values. In general, these values will be a particular realization of some random vector $\vec{\Theta}$. 

As an example, the x-coordinate of any of a particular momentum wheel will conform to some probability distribution, hopefully some distribution with very low variance, but never-the-less random at some level. So if this x-coordinate of many identically manufactured satellites were to be plotted in a histogram, the histogram would mimic the shape of the probability distribution of this x-coordinate.

The random nature of $\vec{\Theta}$ is what is meant by **random in ensemble**.

The random vector $\vec\Theta$ may contain elements for which a set of measurements, $\vec{m}$, can be made, thereby reducing the variance of their individual distributions. That is, the distribution for $\vec\Theta$ would become $\text{d}F_{\vec\Theta|\vec{m}}(\vec\theta|\vec{m})$. Indeed, if a parameter could be measured with infinite accuracy, the conditional measure would become a delta-function probability measure.

### Conditional Probability and Marginalization

The conditional probability distribution for $\mathcal{E}(t)$ is defined as
$$
\text{d}F_{\mathcal{E}(t)|\vec\Theta}\left[e(t)|\vec\theta\right]
$$


With some model for the probability distribution of $\vec\Theta$ (given also some set of measurements, $\vec{m}$), the probability distribution for $\mathcal{E}(t)$ may be expressed as
$$
\text{d}F_{\mathcal{E}(t)}\left(e(t)\right) = \int \text{d}F_{\mathcal{E}(t)|\vec\Theta}\left[e(t)|\vec\theta\right]\text{d}F_{\vec\Theta|\vec{m}}(\vec\theta|\vec{m})
$$

### Time dependence of parameters

No time dependence has been assigned to $\vec\theta$.