import get_data
import pandas as pd
from lstm_model import prepare_data, train_lstm_model


def test_data_retrieval():
    # Test yfinance data retrieval
    yfinance_data = get_data.fetch_yfinance_data()

    if yfinance_data is not None and not yfinance_data.empty:
        print("Yfinance data successfully retrieved!")
    else:
        print("Error retrieving yfinance data.")

    # Test Alpha Vantage data retrieval
    alpha_vantage_data = get_data.fetch_alpha_vantage_data()

    if alpha_vantage_data is not None and not alpha_vantage_data.empty:
        print("Alpha Vantage data successfully retrieved!")
    else:
        print("Error retrieving Alpha Vantage data.")

    return yfinance_data, alpha_vantage_data


def test_date_alignment():
    yfinance_data = get_data.fetch_yfinance_data()
    alpha_vantage_data = get_data.fetch_alpha_vantage_data()

    # Convert the 'feed' column into a list of dictionaries
    feed_data = alpha_vantage_data['feed'].apply(pd.Series)

    # Check if 'time_published' exists in feed_data columns
    if 'time_published' in feed_data.columns:
        alpha_vantage_data['timestamp'] = feed_data['time_published']

        # Convert timestamp to date format
        alpha_vantage_data['date'] = alpha_vantage_data['timestamp'].apply(lambda x: x.split("T")[0])

        # Check if dates in both datasets match
        common_dates = set(yfinance_data.index).intersection(set(alpha_vantage_data['date']))

        assert len(common_dates) > 0, "No common dates found between the datasets!"
        print(f"Common dates between the datasets: {common_dates}")

    else:
        raise ValueError("The 'time_published' key was not found in the Alpha Vantage dataset.")


if __name__ == "__main__":
    yfinance_data, alpha_vantage_data = test_data_retrieval()

    # If you want to inspect the first few rows of the data:
    print("\nYfinance Data Preview:")
    print(yfinance_data.head())

    print("\nAlpha Vantage Data Preview:")
    print(alpha_vantage_data.head())

    test_date_alignment()


