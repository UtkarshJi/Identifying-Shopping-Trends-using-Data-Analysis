
# Shopping Data Analysis & Customer Segmentation

This project performs analysis on a shopping dataset to derive insights about sales trends, customer behavior, and product performance. The key techniques used include data preprocessing, clustering, market basket analysis, and visualization.

## Features:
1. **Data Preprocessing**: 
   - Handling missing values
   - Calculating total sales for each transaction
2. **Sales Analysis**: 
   - Top products by sales
   - Monthly sales trend
3. **Customer Segmentation**: 
   - KMeans clustering to segment customers based on total sales and quantity purchased
4. **Market Basket Analysis**: 
   - Using Apriori algorithm to find frequent itemsets and association rules

## Libraries Used:
- `pandas` for data manipulation
- `numpy` for numerical operations
- `matplotlib` and `seaborn` for data visualization
- `sklearn` for machine learning models (KMeans, StandardScaler)
- `mlxtend` for Market Basket Analysis (Apriori algorithm and Association Rules)

## Installation:
To run this project locally, you'll need Python 3.x and the following libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn mlxtend
```

## Dataset:
The dataset (`shopping_data.csv`) should be placed in the project directory. This dataset contains shopping transactions with columns like `InvoiceDate`, `Description`, `Quantity`, `UnitPrice`, and `CustomerID`.

## Code Walkthrough:

1. **Data Preprocessing**:
   - The data is loaded and missing values are removed.
   - The `TotalSales` for each transaction is calculated as the product of `Quantity` and `UnitPrice`.

2. **Sales Analysis**:
   - The top 10 products by total sales are displayed.
   - Monthly sales trends are visualized with a line plot.

3. **Customer Segmentation**:
   - The data is scaled using `StandardScaler`.
   - KMeans clustering is applied to segment customers based on total sales and quantity purchased.
   - A scatter plot is created to visualize customer clusters.

4. **Market Basket Analysis**:
   - A binary matrix is created to represent the items purchased in each transaction.
   - Apriori algorithm is used to find frequent itemsets, and association rules are derived.
   - A scatter plot visualizes the association rules with metrics like support, confidence, and lift.

## Visualizations:
- **Monthly Sales Trend**: Line plot showing the total sales over time.
- **Top 10 Products by Sales**: Bar chart showing the products that contribute the most to sales.
- **Customer Segmentation**: Scatter plot displaying customer clusters.
- **Correlation Heatmap**: Heatmap showing correlations between quantity, unit price, and total sales.
- **Association Rules**: Scatter plot visualizing the association rules with metrics such as support, confidence, and lift.

## Example Outputs:
- **Top 10 Products by Sales**:
    ```
    Description        TotalSales
    Product A          5000
    Product B          4000
    ...
    ```

- **Customer Segments**:
    ```
    Cluster  TotalSales  Quantity
    0        15000       200
    1        12000       150
    ...
    ```

- **Association Rules**:
    ```
    Rule 1: {Product A} -> {Product B} (Lift: 1.5)
    Rule 2: {Product C} -> {Product D} (Lift: 2.0)
    ...
    ```

## Usage:
Run the script after placing the `shopping_data.csv` file in the directory. The output will include statistical summaries, visualizations, and insights about customer behavior and product sales.

## License:
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
