import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():

    # Step 1: Load the Excel file
    file_path = "your_data.xlsx"  # Replace with your file path
    data = pd.read_excel(file_path)

    # Step 2: Inspect the data
    print("Missing values in each column:")
    print(data.isnull().sum())

    print("\nData types:")
    print(data.dtypes)

    print("\nBasic statistics:")
    print(data.describe())

    # Step 3: Visualize feature distributions
    plt.figure(figsize=(15, 10))
    for i, column in enumerate(data.columns[:-1]):  # Exclude the response column
        plt.subplot(3, 4, i + 1)
        sns.histplot(data[column], kde=True)
        plt.title(column)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15, 10))
    for i, column in enumerate(data.columns[:-1]):  # Exclude the response column
        plt.subplot(3, 4, i + 1)
        sns.boxplot(y=data[column])
        plt.title(column)
    plt.tight_layout()
    plt.show()

    # Step 4: Calculate and visualize correlations
    corr_matrix = data.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation Heatmap")
    plt.show()

    # Step 5: Test correlation of each feature to the response
    response_column = data.columns[-1]

    correlation_with_response = data.corr()[response_column].drop(response_column)

    print("Correlation of each feature with the response:")
    print(correlation_with_response)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=correlation_with_response.index, y=correlation_with_response.values)
    plt.title("Correlation of Features with Response")
    plt.xlabel("Features")
    plt.ylabel("Correlation")
    plt.xticks(rotation=45)
    plt.show()


if __name__ == "__main__":
    main()
