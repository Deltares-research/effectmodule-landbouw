import pandas as pd


def databasesplit(split_area, data, attribute, header):
    #    split_area=split_df
    #    data=cropdata
    # The relevant crop that will be split are filtered from the data in xml file
    n = split_area.Crop.unique()
    split_crop = data.loc[data.Crop.isin(n)].copy()

    split_crop["lookup"] = (
        split_crop[attribute[2]].map(str) + "_" + split_crop[attribute[3]].map(str)
    )
    split_area["lookup"] = (
        split_area[attribute[2]].map(str) + "_" + split_area[attribute[3]].map(str)
    )

    # number of fractions for each crop
    fraction = (
        split_area.drop(["Crop", "Area", "lookup"], axis=1).columns
    ).values.tolist()

    # New columns (number depends on the number of fractions) are added to the database.
    # The columns are filled based on the specified fraction defined for each crop and area
    split_crop = pd.merge(
        split_crop,
        split_area.iloc[:, 2:],
        how="left",
        left_on="lookup",
        right_on="lookup",
    )
    split_crop = split_crop.dropna(how="any")

    # Fraction of areas are multiplied by area
    split_crop[fraction] = split_crop[fraction].multiply(
        split_crop["areaal"], axis="rows"
    )
    split_crop = split_crop.drop(columns=["lookup"])

    # The new areas (split) column are melt as row in the dataframe
    split_crop_melt = pd.melt(
        split_crop, id_vars=header, var_name="CropSplit", value_name="AreaSplit"
    )

    # The new crop is renamed as new number = old number*100 + fraction.
    # For example potato (crop number 3) that have 4 fractions will have such new names: 301, 302, 303, 304
    split_crop_melt["CropSplit"] = split_crop_melt["CropSplit"].apply(pd.to_numeric)
    split_crop_melt["Crop"] = (
        split_crop_melt["Crop"] * 100 + split_crop_melt["CropSplit"]
    )

    # The column with old area values are replaced with new area (with fraction) values.
    split_crop_melt = split_crop_melt.drop(columns=["CropSplit", "areaal"])
    split_crop_melt = split_crop_melt.rename(columns={"AreaSplit": "areaal"})

    # The rows in data from xml file which contains the value before splitting are removed
    data["lookup"] = data["Crop"].map(str) + "_" + data["Area"].map(str)
    #    data=data[~data['lookup'].isin(split)]
    data = data.drop(columns=["lookup"])

    # The new rows which contains the value after splitting are added
    data = pd.concat(
        [data, split_crop_melt]
    )  # data.append(split_crop_melt, ignore_index=False)
    data = data.sort_values(by=attribute, axis=0, ascending=True)
    data = data[header]

    return data


def apply_crop_splitting(
    cropdata: pd.DataFrame,
    sub_crops: pd.DataFrame,
) -> pd.DataFrame:
    """
    Split crops into sub-crops according to the specified fractions.

    Parameters
    ----------
    cropdata : pd.DataFrame
        Crop database containing crop and area information.
    sub_crops : pd.DataFrame
        DataFrame containing the crop splitting fractions.

    Returns
    -------
    pd.DataFrame
        Crop database with the specified crops split into sub-crops.
    """
    cropdata = cropdata.copy()
    split_df = sub_crops.copy()

    # Rename columns to the format expected by databasesplit
    cropdata = cropdata.rename(
        columns={
            "landgebruik": "Crop",
            "regio": "Area",
        }
    )

    # Convert fruit nurseries (crop 21) to fruit (crop 9)
    cropdata.loc[cropdata["Crop"] == 21, "Crop"] = 9

    header = list(cropdata.columns)
    # compare with: header = list(cropdata.columns.values)

    # Prepare crop splitting information
    split_df = split_df.reset_index()
    split_df = split_df.rename(columns={"Split_Crop": "Crop"}).astype(float)

    if split_df.empty:
        print("No splitting!")
        return cropdata

    attributes = [
        "Scenario",
        "Year",
        "Crop",
        "Area",
    ]

    cropdata = databasesplit(
        split_df,
        cropdata,
        attributes,
        header,
    )

    # Remove original crop areas after splitting
    for crop in split_df["Crop"].unique():
        cropdata.loc[
            cropdata["Crop"] == crop,
            "areaal",
        ] = 0

    cropdata = cropdata[cropdata["areaal"] != 0]

    return cropdata
