# Fundamentals of Statistics Unit 2 Lecture 3 Notes

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import *
import warnings
from IPython.display import display, HTML

warnings.filterwarnings('ignore')

# Trinity of Statistical Inference

  # 1. Estimation
  # 2. Confidence Intervals

# Estimating the mean number of people who arrive at a Spenny's on day between 12pm and 1pm by 
# recording patrons for 50 consecutive days (n=50). Experiment is repeated 1000 times.
# Poisson may not be the right model, since I'm getting number of people who arrived on a day
# rather than time between arrivals. But, that doesn't hurt the example.

n = 50 
mean = 25
experiment_count = 1000
samples = np.array([np.random.poisson(mean, n) for i in range(experiment_count)])
sample_means = np.mean(samples, axis=1)
est_stds = \
    np.sqrt(
        1 / n * np.sum(np.square([samples[i] - sample_means[i] for i in range(experiment_count)]), 
                       axis=1)
    ) 

fig = plt.figure(figsize=(20,4))
fig.suptitle("Recording Spenny's patrons 12pm-1pm for 50 days (4/1000 experiment iterations)", fontsize=24)
for i in range(4):
    plt.subplot(1, 4, i+1)
    if i == 0:
        plt.xlabel('patrons', fontsize=14)
        plt.ylabel('frequency', fontsize=14)
    sns.distplot(samples[i], kde=False, bins=15)
plt.show()

# a = 0.05, Phi^-1(0.05/2) = -1.96  if a=0.05, Confidence level = 0.95. We expect to see roughly 5% of 
# CI's fail to capture the true mean. The proportion converges to at most 0.05 as n -> infinity.

proportion_bad_CIs = (np.sum(sample_means - 1.96 * est_stds / np.sqrt(n) > mean) \
    + np.sum(sample_means + 1.96 * est_stds / np.sqrt(n) < mean)) \
    / experiment_count

plt.title('Mean Estimations', fontsize=18)
plt.xlabel('experiment iteration estimated mean', fontsize=12)
plt.ylabel('density', fontsize=12)
ax = sns.distplot(sample_means, bins=50)
x, y = ax.get_lines()[-1].get_data()
mask = x < mean - 1.96 * np.mean(est_stds) / np.sqrt(n) 
x_1, y_1 = x[mask], y[mask]
ax.fill_between(x_1, y1=y_1, facecolor='red', alpha=0.8)
mask = x > mean + 1.96 * np.mean(est_stds) / np.sqrt(n)
x_2, y_2 = x[mask], y[mask]
ax.fill_between(x_2, y1=y_2, facecolor='red', alpha=0.8)
plt.show()

ci_l = sample_means[0] - 1.96 * est_stds[0] / np.sqrt(n) 
ci_r = sample_means[0] + 1.96 * est_stds[0] / np.sqrt(n)
captured = ci_l < mean and ci_r > mean
    
cap_or_not = 'captured' if captured else 'did not capture'
display(HTML('<h4>The red area roughly shows which mean estimations had CI\'s that failed to capture the true mean at 95% confidence</h4>'))
display(HTML(f'<h4> Proportion of CIs that fail to capture the mean at $alpha = 0.05$: {proportion_bad_CIs}</h4>\n' \
            + '(the proportion will converge to at most 0.05 as n goes to infinity)'))
display(HTML(f'<h4> We {cap_or_not} the mean (25) in our first experiment iteration\n' \
             f'with our confidence interval [{ci_l:.2f}, {ci_r:.2f}]</h4>'))

  # 3. Hypothesis Testing             

# Statistical Modeling

  # Simplification

# Working backwards from a Poisson here because it's easier.
mean =  1.5
n = 5000
pois = pd.DataFrame({'P(X = x)':np.random.poisson(mean, 5000)})['P(X = x)'].value_counts() / n
pois.loc[7] += pois.loc[7:pois.shape[0] - 1].sum()
pois = pois.loc[:7]
pois.rename(index={0:'p0', 1:'p1', 2:'p2', 3:'p3', 4:'p4', 5:'p5', 6:'p6', 7:'p>=7'}, inplace=True)
pois = pois.sort_index()

display(HTML('<h4> $X \in \{1,2,...,\geq7\}$, 8 parameters </h4>'))
display(HTML(pd.DataFrame(pois).transpose().to_html()))
display(HTML('<h4> Poisson RV, 1 parameter </h4>'))
display(HTML('$Poisson(\lambda = 1.5)$'))
display(HTML('<h4> Same Distribution </h4>'))

plt.figure(figsize=(6,4))
plt.title('Number of Siblings of Individuals',fontsize=16)
sns.barplot(pois.index, pois, palette=sns.cubehelix_palette(8, start=.5, rot=-.75))
plt.ylabel('proportion', fontsize=14)
plt.xlabel('number of siblings', fontsize=14)
plt.xticks([i for i in range(8)], [f'{i}' if i < 7 else f'>={i}' for i in range(8)])

plt.show()


# Parametric, nonparametric and semiparametric models

plt.figure(figsize=(6,6))
plt.title('Real distribution, within a family, within all distributions')
E = plt.Circle((0, 0), radius=1, fc='#99c24d')
P_theta = plt.Circle((0.4, 0.35), radius = 0.3, fc='#048ba8')
P = plt.Circle((0.25, 0.28), radius=0.02, fc='w')
plt.gca().add_patch(E)
plt.gca().add_patch(P_theta)
plt.gca().add_patch(P)
plt.text(0.29, 0.26, 'P', color='w', fontsize=14, fontweight=900)
plt.text(-0.7, -0.53, 'E', color='#a9e27d', fontsize=128, fontweight=900)
plt.text(0.21, 0.38, 'Ptheta', color='#84dbf8', fontsize=18, fontweight=900)
plt.axis('scaled')
plt.yticks([])
plt.xticks([])
plt.show()

# Statistical Model Examples

  # Linear Regression Model

# I'm not 100% clear on the concept of linear regression presented above. My sim here
# represents a few linear regression models - one true and one estimated.

# X from 0 to 10 represents hours 7am to 5pm, open to close. (weird hours, sure)
X = np.arange(0, 11, 1)

# unknown parameters in the true model
beta_0 = 0.5
beta_1 = 5
# beta_2 is an unknown parameter that gives a slight curve, but we're ignoring because this domain
# of X has f(X) modeled well enough linearly
beta_2 = -0.5 

# I usually simulate error with a variance > 1. I'm not sure why class slides have N(0, 1) for error,
# but that will probably be explained later in class.
error = np.random.normal(0, 2, 11)
Y = beta_0 + beta_1 * X + beta_2 * X**2 + error

# This is a closed form solution to finding beta_1 and beta_0 without bias
# in our desired estimation domain 7am to 1pm
est_beta_1 = np.sum((X[:7] - np.mean(X[:7])) * (Y[:7] - np.mean(Y[:7]))) / np.sum((X[:7] - np.mean(X[:7]))**2)
est_beta_0 = np.mean(Y[:7]) - est_beta_1 * np.mean(X[:7])

figure(figsize=(12,3))
plt.suptitle('Spenny\'s Customer Population from:')
plt.subplot(1, 2, 1)
plt.title('7am to 1pm', fontsize=14)
sns.scatterplot(X[:7], Y[:7])
sns.lineplot(X[:7], est_beta_0 + est_beta_1 * X[:7], color='green')
plt.xticks(X[:7], ['7am', '8am', '9am', '10am', '11am', '12pm', '1pm'])

plt.subplot(1, 2, 2)
plt.title('7am to 5pm', fontsize=14)
sns.scatterplot(X, Y)
sns.lineplot(X, est_beta_0 + est_beta_1 * X, color='green')
plt.xticks(X, ['7am', '8am', '9am', '10am', '11am', '12pm', '1pm', '2pm', '3pm', '4pm', '5pm'])
plt.ylim(0, 19)
plt.show()

table = pd.DataFrame(
    {'Actual': [beta_0, beta_1, beta_2], 'Estimated': [round(est_beta_0,2), round(est_beta_1,2), 0]})
table.rename(index={0:'beta_0', 1:'beta_1', 2:'beta_2'}, inplace=True)
print(table)
print('\nThe estimated parameters are off because outside of the domain of our estimation,\n f(x) looks very different.')
print('You could say, however, that customer population is modeled well with a simple\n linear model between 7am and 1pm.')
print('There will be many times when you don\'t know what\'s beyond 1pm and this model works\n fine anyway, even if adding a higher order parameter would predict a little better.')
print('This simpler approach is often useful for explaining how one variable affects another,\n where adding higher order parameters convolutes the message.')
print('If you want to predict # of customers rather than figure out the relationship between\n hours and customers, adding beta_2 * X^2 is a good idea.')


# Cox Proportional Hazard Model

# Identifiability

  # Injectivity
  # Identifiability of Statistical Models

