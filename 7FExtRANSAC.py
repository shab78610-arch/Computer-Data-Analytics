import numpy as np
from sklearn.linear_model import RANSACRegressor
import matplotlib.pyplot as plt
# Generate some noisy data points
np.random.seed(0)
x = np.random.uniform(0, 10, 100)
y = 2 * x+1 + np.random.normal(0, 1, 100)
#Add Outliers
outliers_index = np.random.choice(100, 20, replace=False)
y[outliers_index] +=10 * np.random.normal(0, 1, 20)
# Stack the points for RANSAC
data = np.vstack((x,y)).T
# Intialize RANSAC model (using sklearn)
ransac = RANSACRegressor()
# Fit the model
ransac.fit(data[:, 0]. reshape(-1,1), data [:, 1])
# Extract the inliers and outliers
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
# Extract the line parameters
line_slope = ransac.estimator_.coef_[0]
line_intercept = ransac.estimator_.intercept_
# Plot the data points
plt.scatter(data[inlier_mask][:, 0], data[inlier_mask][:, 1], c='b' , label='Inliers')
plt.scatter(data[outlier_mask][:, 0], data[outlier_mask][:, 1], c='r', label='Outliers')
# Plot the Fitted Line
plt.plot(x,line_slope * x + line_intercept, color='g', label='RANSAC line')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()
