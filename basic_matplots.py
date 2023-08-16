import matplotlib.pyplot as plt
import numpy as np
from main import *

plt.rcParams['figure.figsize'] = (18, 7)

color = plt.cm.copper(np.linspace(0, 1, 40))
data['Methods'].value_counts().head(40).plot.bar(color=color)
plt.title('Most Popular Methods by the Users', fontsize=20)
plt.show()

plt.rcParams['figure.figsize'] = (18, 7)

color = plt.cm.copper(np.linspace(0, 1, 40))
data['month'].value_counts().head(40).plot.bar(color='cyan')
plt.title('Most Popular Months of Logins', fontsize=20)
plt.show()

plt.rcParams['figure.figsize'] = (18, 7)

color = plt.cm.copper(np.linspace(0, 1, 40))
data['day'].value_counts().head(40).plot.bar(color='tomato')
plt.title('Most Popular Days of Logins', fontsize=20)
plt.show()

plt.rcParams['figure.figsize'] = (18, 7)

color = plt.cm.Wistia(np.linspace(0, 1, 40))
data['Status'].value_counts().head(40).plot.bar(color='seagreen')
plt.title('Most Popular statuses for the Users', fontsize=20)
plt.show()
