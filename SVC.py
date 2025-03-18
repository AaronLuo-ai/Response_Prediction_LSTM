import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


def main():
    # Load the modified Excel file
    file_path = "C:\\Users\\aaron.l\\Documents\\radiomics_features_modified.xlsx"
    db = pd.read_excel(file_path)

    # Step 1: Prepare features (X) and labels (y)
    # Exclude 'Response', 'Patient_ID', and 'Session' columns for features
    X = db.drop(columns=["response", "Patient_ID", "Session"])
    y = db["response"]  # Labels are the 'Response' column

    # Step 2: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42
    )

    # Step 3: Train an SVC model
    svc_model = SVC(
        kernel="rbf", random_state=42
    )  # You can change the kernel if needed
    svc_model.fit(X_train, y_train)

    # Step 4: Evaluate the model on the test data
    y_pred = svc_model.predict(X_test)

    # Print evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    response_counts = db["response"].value_counts()

    # Print the results
    print(response_counts)


if __name__ == "__main__":
    main()
