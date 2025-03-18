import pandas as pd


def main():

    # Step 1: Load the Excel files
    radiomics_features = pd.read_excel(
        "C:\\Users\\aaron.l\\Documents\\nrrd_images_masks_simple\\radiomics_features.xlsx"
    )
    db = pd.read_excel("C:\\Users\\aaron.l\\Documents\\db_20241213.xlsx")

    # Step 1: Extract Patient_ID and Session from cnda_session_label in the db DataFrame
    db[["Patient_ID", "Session"]] = db["cnda_session_label"].str.extract(
        r"cnda_p(\d+)_MR(\d+)"
    )

    # Step 2: Add 'p' to Patient_ID in db to match the format in radiomics_features
    db["Patient_ID"] = "p" + db["Patient_ID"]
    db["Session"] = "MR" + db["Session"]

    # Step 3: Ensure Patient_ID and Session in radiomics_features are strings
    radiomics_features["Patient_ID"] = radiomics_features["Patient_ID"].astype(str)
    radiomics_features["Session"] = radiomics_features["Session"].astype(str)

    # Step 4: Create a dictionary to map (Patient_ID, Session) to AJCC Stage grouping
    response_map = db.set_index(["Patient_ID", "Session"])[
        "AJCC Stage grouping "
    ].to_dict()

    # Step 7: Add the response column to radiomics_features
    radiomics_features["response"] = None  # Initialize the column with None

    # Step 8: Check each row in radiomics_features and update the response column
    for index, row in radiomics_features.iterrows():
        key = (row["Patient_ID"], row["Session"])
        if key in response_map:
            radiomics_features.at[index, "response"] = response_map[key]

    radiomics_features["response"] = pd.to_numeric(
        radiomics_features["response"], errors="coerce"
    )  # Convert to numeric, non-numeric becomes NaN
    radiomics_features = radiomics_features.dropna(
        subset=["response"]
    )  # Drop rows where 'Response' is NaN

    # Step 2: Convert the 'Response' column values
    radiomics_features["response"] = radiomics_features["response"].apply(
        lambda x: 1 if x in [0, 1] else 0
    )

    # Step 3: Save the modified DataFrame to a new Excel file
    output_path = "C:\\Users\\aaron.l\\Documents\\radiomics_features_modified.xlsx"
    radiomics_features.to_excel(output_path, index=False)


if __name__ == "__main__":
    main()
