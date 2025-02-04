import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules

sns.set(style="whitegrid")

data = pd.read_csv('shopping_data.csv')

data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
data.dropna(inplace=True)

data['TotalSales'] = data['Quantity'] * data['UnitPrice']

data.info()
print("\nSummary Statistics:")
print(data.describe())

top_products = data.groupby('Description')['TotalSales'].sum().nlargest(10)
print("\nTop 10 Products by Sales:")
print(top_products)

monthly_sales = data.resample('M', on='InvoiceDate')['TotalSales'].sum()
print("\nMonthly Sales Trend:")
print(monthly_sales)

customer_data = data.groupby('CustomerID').agg({'TotalSales': 'sum', 'Quantity': 'sum'}).reset_index()
scaler = StandardScaler()
scaled_data = scaler.fit_transform(customer_data[['TotalSales', 'Quantity']])
kmeans = KMeans(n_clusters=4, random_state=42)
customer_data['Cluster'] = kmeans.fit_predict(scaled_data)

print("\nCustomer Segments:")
print(customer_data.groupby('Cluster').mean())

plt.figure(figsize=(10, 6))
monthly_sales.plot(kind='line', marker='o', color='b')
plt.title('Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.show()

plt.figure(figsize=(10, 6))
top_products.plot(kind='bar', color='skyblue')
plt.title('Top 10 Products by Sales')
plt.xlabel('Product')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='TotalSales', y='Quantity', hue='Cluster', data=customer_data, palette='viridis')
plt.title('Customer Segmentation')
plt.xlabel('Total Sales')
plt.ylabel('Quantity Purchased')
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(data[['Quantity', 'UnitPrice', 'TotalSales']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

basket = data.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().fillna(0)
basket = basket.applymap(lambda x: 1 if x > 0 else 0)

frequent_itemsets = apriori(basket, min_support=0.05, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

print("\nTop 5 Association Rules:")
print(rules.head())

plt.figure(figsize=(10, 6))
sns.scatterplot(x='support', y='confidence', size='lift', data=rules, hue='lift', palette='viridis')
plt.title('Association Rules (Market Basket Analysis)')
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.show()
