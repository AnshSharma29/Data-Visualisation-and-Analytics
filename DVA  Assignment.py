#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Q1. Write a program to create a DataFrame have E-commerce data and perform selection of row/column using loc() and iloc()

import pandas as pd

# Read CSV file into DataFrame with specified encoding
ecommerce_df = pd.read_csv('ecommerce_data.csv', encoding='latin1')

# Display the DataFrame
print("E-commerce data:")
print(ecommerce_df)
print()

# using loc()
print("Selection using loc():")
# Selecting rows where UnitPrice > 500
print("Rows where UnitPrice > 500:")
print(ecommerce_df.loc[ecommerce_df['UnitPrice'] > 500])
print()

# Selecting specific columns and rows using loc()
print("Selecting specific columns and rows using loc():")
# Selecting rows where Quantity is greater than 1 and only displaying InvoiceNo and Total columns
print(ecommerce_df.loc[ecommerce_df['Quantity'] > 1, ['InvoiceNo', 'Total']])
print()

# Selection using iloc()
print("Selection using iloc():")
# Selecting rows at index 1 and 3
print(ecommerce_df.iloc[[1, 3]])
print()

# Selecting specific rows and columns using iloc()
print("Selecting specific rows and columns using iloc():")
# Selecting rows at index 0, 2 and only displaying Product and Quantity columns
print(ecommerce_df.iloc[[0, 2], [1, 3]])


# In[2]:


#Q2. Create a Series object S5 containing numbers. Write a program to store the square 
#    of the series values in object S6. Display S6’s values which are >15.

import pandas as pd

# Series object S5 containing numbers
S5 = pd.Series([1, 2, 3, 4, 5])

# square of the series values in object S6
S6 = S5 ** 2

# Display S6's values which are >15
print("S6's values > 15:")
print(S6[S6 > 15])


# In[4]:


#Q3. Write a program to fill all missing values in a DataFrame with zero. 


import pandas as pd

# Read CSV file into DataFrame with specified encoding
ecommerce_df = pd.read_csv('ecommerce_data.csv', encoding='latin1')

# Fill missing values with zero
ecommerce_df_filled = ecommerce_df.fillna(0)

# Display the filled DataFrame
print("DataFrame with missing values filled with zero:")
print(ecommerce_df_filled)


# In[6]:


#Q4 Program for combining DataFrames using concat(), join(),merge() 

import pandas as pd

# Read CSV file into DataFrame with specified encoding
ecommerce_df = pd.read_csv('ecommerce_data.csv', encoding='latin1')

# Let's create another DataFrame for demonstration purposes
# This DataFrame will contain sample customer data
customer_data = pd.DataFrame({
    'CustomerID': [17850, 13047, 12583, 13748, 15100],
    'Name': ['John', 'Jane', 'Michael', 'Emily', 'David'],
    'Country': ['United Kingdom', 'Germany', 'France', 'Italy', 'Spain']
})

# Concatenating DataFrames using concat()
concatenated_df = pd.concat([ecommerce_df, customer_data], axis=1)

# Display the concatenated DataFrame
print("Concatenated DataFrame:")
print(concatenated_df)
print()

# Merging DataFrames using merge()
merged_df = pd.merge(ecommerce_df, customer_data, on='CustomerID', how='inner', suffixes=('_ecommerce', '_customer'))

# Display the merged DataFrame
print("Merged DataFrame:")
print(merged_df)
print()

# Joining DataFrames using join()
# Let's set 'CustomerID' as the index for both DataFrames
ecommerce_df.set_index('CustomerID', inplace=True)
customer_data.set_index('CustomerID', inplace=True)

# Joining the DataFrames
joined_df = ecommerce_df.join(customer_data, how='inner', lsuffix='_ecommerce', rsuffix='_customer')

# Resetting the index to have 'CustomerID' as a regular column
joined_df.reset_index(inplace=True)

# Display the joined DataFrame
print("Joined DataFrame:")
print(joined_df)


# In[7]:


#Q5 Write a program to draw bar graph for the following data for the Medal tally of 
#Olympic games 
#Gold Silver Bronze Total 
# 26   20     20     66 

import matplotlib.pyplot as plt

# Data
categories = ['Gold', 'Silver', 'Bronze', 'Total']
medals = [26, 20, 20, 66]

# Plotting the bar graph
plt.figure(figsize=(8, 6))
plt.bar(categories, medals, color=['gold', 'silver', 'brown', 'orange'])
plt.xlabel('Medal Type')
plt.ylabel('Number of Medals')
plt.title('Medal Tally of Olympic Games')
plt.show()


# In[8]:


#Q6 Implementing Line plot, Dist plot, Lmplot, Count plot using Seaborn library

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Create a sample DataFrame
data = pd.DataFrame({
    'X': range(10),
    'Y': [1, 3, 2, 5, 4, 7, 6, 9, 8, 10],
    'Category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
})

# Line plot
plt.figure(figsize=(8, 6))
sns.lineplot(data=data, x='X', y='Y')
plt.title('Line Plot')
plt.show()

# Dist plot
plt.figure(figsize=(8, 6))
sns.distplot(data['Y'])
plt.title('Distribution Plot')
plt.show()

# Lmplot
plt.figure(figsize=(8, 6))
sns.lmplot(data=data, x='X', y='Y', hue='Category', markers=['o', 'x'])
plt.title('Lmplot')
plt.show()

# Count plot
plt.figure(figsize=(8, 6))
sns.countplot(data=data, x='Category')
plt.title('Count Plot')
plt.show()


# In[9]:


#Q7 Create a DataFrame namely aid that stores aid (Toys, books, uniform, shoes) by 
#   NGO’s for different states. Write a program to display the aid for: - 
#   (a) Books and Uniforms only 
#   (b) Shoes only 

import pandas as pd

# DataFrame for aid by NGOs for different states
aid_data = {
    'State': ['State1', 'State2', 'State3', 'State4'],
    'Toys': [100, 150, 120, 180],
    'Books': [200, 250, 220, 270],
    'Uniform': [150, 180, 160, 190],
    'Shoes': [80, 100, 90, 120]
}

aid = pd.DataFrame(aid_data)

# Display aid for Books and Uniforms only
print("Aid for Books and Uniforms only:")
print(aid[['State', 'Books', 'Uniform']])

# Display aid for Shoes only
print("\nAid for Shoes only:")
print(aid[['State', 'Shoes']])


# In[10]:


#Q8 Create a DataFrame df having Name, Gender, Position, City, Age, 
#   Projects. Write a program to summarize how many projects are being 
#   handled by each position for each city? Use pivot() 


import pandas as pd

# DataFrame
data = {
    'Name': ['John', 'Jane', 'Mike', 'Emily', 'David'],
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
    'Position': ['Manager', 'Developer', 'Manager', 'Developer', 'Developer'],
    'City': ['New York', 'Chicago', 'New York', 'Chicago', 'Los Angeles'],
    'Age': [35, 28, 40, 32, 45],
    'Projects': [5, 8, 6, 7, 9]
}

df = pd.DataFrame(data)

# Summarize projects handled by each position for each city
summary = df.pivot_table(index=['Position', 'City'], aggfunc='sum', values='Projects')

print("Summary of projects handled by each position for each city:")
print(summary)


# In[11]:


#Q9 Marks is a list that stores marks of a student in 10-unit test. Write a program to 
#   plot Line chart for the student’s performance in these 10 tests.

import matplotlib.pyplot as plt

# Example marks list 
marks = [85, 90, 92, 88, 86, 89, 91, 87, 90, 93]

# x-axis labels (test numbers)
test_numbers = range(1, len(marks) + 1)

# Plotting the line chart
plt.figure(figsize=(8, 6))
plt.plot(test_numbers, marks, marker='o', linestyle='-')
plt.title("Student's Performance in 10 Tests")
plt.xlabel("Test Number")
plt.ylabel("Marks")
plt.grid(True)
plt.xticks(test_numbers)  # Set x-axis ticks to match test numbers
plt.tight_layout()
plt.show()


# In[15]:


#Q10 Write a program to plot a horizontal bar chart from the height of some students.

import matplotlib.pyplot as plt

# Student names
students = ['John', 'Emma', 'Michael', 'Sophia', 'William']

# Heights of students (in inches)
heights = [65, 63, 70, 67, 72]

# Plotting the horizontal bar chart
plt.figure(figsize=(8, 6))
plt.barh(students, heights, color='skyblue')
plt.xlabel('Height (in inches)')
plt.ylabel('Students')
plt.title('Heights of Students')
plt.grid(axis='x')  # Show gridlines only along the x-axis
plt.tight_layout()
plt.show()


# In[16]:


#Q11  Write a program to implement Covariance.

def covariance(X, Y):
    n = len(X)
    if n != len(Y):
        raise ValueError("X and Y must have the same length")

    # Calculate the means of X and Y
    mean_X = sum(X) / n
    mean_Y = sum(Y) / n

    # Calculate the covariance
    cov = sum((X[i] - mean_X) * (Y[i] - mean_Y) for i in range(n)) / n

    return cov

# Example data
X = [1, 2, 3, 4, 5]
Y = [2, 3, 4, 5, 6]

# Calculate covariance
cov_xy = covariance(X, Y)
print("Covariance between X and Y:", cov_xy)


# In[ ]:


#Q12 Exploratory Analysis start


