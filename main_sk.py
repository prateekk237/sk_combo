import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from itertools import permutations

# Helper function for generating surrounding digits
def surrounding_digits(middle_digit):
    digits = list(range(10))
    middle_index = digits.index(middle_digit)
    before = digits[middle_index - 3:middle_index]
    after = digits[(middle_index + 1):(middle_index + 4)]
    if len(after) < 3:
        after += digits[:(3 - len(after))]
    return before + after

# Function to scrape DSWR numbers for the last 3 months
def scrape_dswr_numbers(start_month, start_year, end_month, end_year):
    results = []
    start_date = datetime(start_year, start_month, 1)
    end_date = datetime(end_year, end_month, 1)
    current_date = start_date

    while current_date <= end_date:
        month = current_date.month
        year = current_date.year
        url = f"https://satta-king-fast.com/desawar/satta-result-chart/ds/?month={month}&year={year}"
        print(f"Scraping URL: {url}")
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to fetch data for {month}/{year}")
            current_date = current_date.replace(day=28) + pd.Timedelta(days=4)
            current_date = current_date.replace(day=1)
            continue

        soup = BeautifulSoup(response.content, "html.parser")
        table_rows = soup.find_all("tr", class_="day-number")
        print(f"Found {len(table_rows)} day-number rows.")

        for row in table_rows:
            day_cell = row.find("td", class_="day")
            day = day_cell.get_text(strip=True) if day_cell else "empty"
            number_cells = row.find_all("td", class_="number")
            dswr_number = number_cells[0].get_text(strip=True) if number_cells and number_cells[0] else "empty"
            results.append((day, dswr_number))

        current_date = current_date.replace(day=28) + pd.Timedelta(days=4)
        current_date = current_date.replace(day=1)

    df = pd.DataFrame(results, columns=["Day", "DSWR Number"])
    return df

# Function to create and train the prediction model
def create_prediction_model(data):
    # Preprocess data (convert "empty" to NaN, then to 0)
    data["DSWR Number"] = pd.to_numeric(data["DSWR Number"], errors="coerce").fillna(0)

    # Create lag features for time series (use the last 7 days to predict the next day's DSWR number)
    for i in range(1, 8):
        data[f"lag_{i}"] = data["DSWR Number"].shift(i)

    # Drop rows with NaN values (first 7 rows will have NaNs due to the lag features)
    data = data.dropna()

    # Define the features and target
    X = data.drop(columns=["Day", "DSWR Number"])  # Features: previous days' DSWR numbers
    y = data["DSWR Number"]  # Target: DSWR number for the next day

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

    return model

# Function to predict the next day's DSWR number
def predict_dswr(model, data):
    # Get the last 7 days of data
    latest_data = data.iloc[-1][1:-1].values.reshape(1, -1)  # Last row, excluding "Day" and "DSWR Number" columns
    print(f"Prediction input shape: {latest_data.shape}")
    # Make the prediction for the next day's DSWR number
    predicted_dswr = model.predict(latest_data)
    return predicted_dswr[0]

# Streamlit app UI
st.title("SK Combinations 👑💰")

# Scrape DSWR data for the last 3 months
today = datetime.today()
start_month = today.month - 2 if today.month - 2 > 0 else 1
start_year = today.year if today.month - 2 > 0 else today.year - 1
end_month = today.month
end_year = today.year

data = scrape_dswr_numbers(start_month, start_year, end_month, end_year)

# Create and train the prediction model
model = create_prediction_model(data)

# Ensure session_state variables are initialized
if "pair1" not in st.session_state:
    st.session_state.pair1 = str(data["DSWR Number"].iloc[-2])  # First number from the last two
if "pair2" not in st.session_state:
    st.session_state.pair2 = str(data["DSWR Number"].iloc[-1])  # Second number from the last two

# Initialize combinations and prediction results if not already initialized
if "predicted_dswr" not in st.session_state:
    st.session_state.predicted_dswr = None
if "all_combinations" not in st.session_state:
    st.session_state.all_combinations = []

# Display input fields for pairs
pair1 = st.text_input("Enter first two-digit pair (e.g., 84):", st.session_state.pair2)
pair2 = st.text_input("Enter second two-digit pair (e.g., 09):", st.session_state.pair1)

# Automatically clear prediction value when combinations are generated
if st.button("Generate Combinations"):
    first_digit = int(pair1[0])
    second_digit = int(pair2[0])
    middle_number = (first_digit + second_digit) % 10

    surrounding = surrounding_digits(middle_number)

    two_digit_combinations = [''.join(p) for p in permutations(map(str, surrounding), 2)]
    double_combinations = [''.join([d, d]) for d in map(str, surrounding)]
    all_two_digit_combinations = sorted(set(two_digit_combinations + double_combinations))

    # Store combinations in session state
    st.session_state.all_combinations = all_two_digit_combinations

    st.markdown(f"**Surrounding digits (excluding {middle_number}):**", unsafe_allow_html=True)
    st.markdown(f"<span style='color:red;'>{', '.join(map(str, surrounding))}</span>", unsafe_allow_html=True)
    st.markdown(f"**All two-digit combinations including doubles ({len(all_two_digit_combinations)} results):**")
    st.markdown(f"<span style='color:green;'>{', '.join(all_two_digit_combinations)}</span>", unsafe_allow_html=True)

# Prediction DSWR button
if st.button("Prediction DSWR"):
    predicted_dswr = predict_dswr(model, data)
    st.session_state.predicted_dswr = predicted_dswr
    #st.markdown(f"**Predicted DSWR Number for Next Day:**")
    #st.markdown(f"<span style='color:green;'>{predicted_dswr}</span>", unsafe_allow_html=True)

# Display stored prediction and combinations
if st.session_state.predicted_dswr is not None:
    st.markdown(f"**Stored Prediction DSWR Number for Next Day:**")
    st.markdown(f"<span style='color:green;'>{st.session_state.predicted_dswr}</span>", unsafe_allow_html=True)

if st.session_state.all_combinations:
    st.markdown(f"**Stored Combinations Results:**")
    st.markdown(f"<span style='color:green;'>{', '.join(st.session_state.all_combinations)}</span>", unsafe_allow_html=True)
