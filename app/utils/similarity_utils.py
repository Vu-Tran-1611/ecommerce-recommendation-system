# Get index by product_id in the dataframe 
def get_product_index(product_id, df):
    try:
        return df[df["product_id"] == product_id].index[0]
    except IndexError:
        raise ValueError(f"Product ID {product_id} not found in the dataset.")


# Get product_ids by indices from the dataframe
def get_product_ids(indices, df):
    return df.iloc[indices]["product_id"].tolist()  
