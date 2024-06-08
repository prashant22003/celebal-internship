#-----------------------------------------------First--------------------------------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load the Iris dataset from seaborn's built-in datasets
iris = sns.load_dataset("iris")

# Display the first few rows of the dataset
iris.head()

#----------------------------------------------Second---------------------------------------------------------------------------

from sklearn.datasets import load_iris
import pandas as pd

# Load the Iris dataset from sklearn
iris_sklearn = load_iris()

# Create a DataFrame from the sklearn dataset
iris_df = pd.DataFrame(data=iris_sklearn.data, columns=iris_sklearn.feature_names)
iris_df['species'] = iris_sklearn.target

# Map target numbers to actual species names
species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
iris_df['species'] = iris_df['species'].map(species_map)

# Display the first few rows of the dataset
iris_df.head()

Data: 

Results         sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)      species
   0                5.1               3.5                1.4               0.2                setosa
   1                4.9               3.0                1.4               0.2                setosa
   2                4.7               3.2                1.3               0.2                setosa
   3                4.6               3.1                1.5               0.2                setosa
   4                5.0               3.6                1.4               0.2                setosa

#------------------------------------------------------Third---------------------------------------------------------------------

import seaborn as sns

# Set the aesthetic style of the plots
sns.set(style="whitegrid")

# Pair Plot
pair_plot = sns.pairplot(iris_df, hue="species", markers=["o", "s", "D"])
pair_plot.fig.suptitle("Pair Plot of Iris Dataset", y=1.02)  # Adjust the title position

# Box Plot
plt.figure(figsize=(10, 6))
box_plot = sns.boxplot(x="species", y="sepal length (cm)", data=iris_df)
box_plot.set_title("Box Plot of Sepal Length by Species")

# Violin Plot
plt.figure(figsize=(10, 6))
violin_plot = sns.violinplot(x="species", y="sepal width (cm)", data=iris_df)
violin_plot.set_title("Violin Plot of Sepal Width by Species")

# Scatter Plot
plt.figure(figsize=(10, 6))
scatter_plot = sns.scatterplot(x="petal length (cm)", y="petal width (cm)", hue="species", style="species", data=iris_df)
scatter_plot.set_title("Scatter Plot of Petal Length vs Petal Width")

# Show all plots
plt.show()


#---------------------------------------------------------------------Fourth---------------------------------------------------------------

# Pair Plot
plt.figure()
sns.pairplot(iris_df, hue="species")
plt.suptitle("Pair Plot of Iris Dataset", y=1.02)  # Adjust the title position

# Box Plot
plt.figure(figsize=(10, 6))
sns.boxplot(x="species", y="sepal length (cm)", data=iris_df)
plt.title("Box Plot of Sepal Length by Species")

# Violin Plot
plt.figure(figsize=(10, 6))
sns.violinplot(x="species", y="sepal width (cm)", data=iris_df)
plt.title("Violin Plot of Sepal Width by Species")

# Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x="petal length (cm)", y="petal width (cm)", hue="species", style="species", data=iris_df)
plt.title("Scatter Plot of Petal Length vs Petal Width")

# Show all plots
plt.show()
