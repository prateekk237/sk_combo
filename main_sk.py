import streamlit as st
from itertools import permutations


# Function to generate digits around a given number
def surrounding_digits(middle_digit):
    digits = list(range(10))  # Digits from 0 to 9
    middle_index = digits.index(middle_digit)

    # Get three digits before the middle digit
    before = digits[middle_index - 3:middle_index]

    # Get three digits after the middle digit (wrap around)
    after = digits[(middle_index + 1):(middle_index + 4)]
    if len(after) < 3:
        after += digits[:(3 - len(after))]

    return before + after


# Streamlit app
st.title("SK Combinations 👑💰")

# Input pairs
pair1 = st.text_input("Enter first two-digit pair (e.g., 84):", "84")
pair2 = st.text_input("Enter second two-digit pair (e.g., 09):", "09")

# Align buttons in a row
col1, col2 = st.columns(2)

with col1:
    generate_button = st.button("Generate Combinations")

with col2:
    if st.button("View Chart"):
        st.markdown("[Open Satta Result Chart](https://satta-king-fast.com/desawar/satta-result-chart/ds/)",
                    unsafe_allow_html=True)

# Generate combinations when the button is clicked
if generate_button:
    # Extract the first digit from each pair
    first_digit = int(pair1[0])
    second_digit = int(pair2[0])

    # Calculate the middle number
    middle_number = (first_digit + second_digit) % 10  # Ensure within 0-9 range

    # Get three digits before and after the middle number
    surrounding = surrounding_digits(middle_number)

    # Generate all two-digit combinations including doubles
    two_digit_combinations = [''.join(p) for p in permutations(map(str, surrounding), 2)]
    double_combinations = [''.join([d, d]) for d in map(str, surrounding)]
    all_two_digit_combinations = sorted(set(two_digit_combinations + double_combinations))

    # Display results
    st.markdown(f"**<span style='color:red;'>Surrounding digits (excluding {middle_number}):</span>**",
                unsafe_allow_html=True)
    st.markdown(f"<span style='color:red;'>{', '.join(map(str, surrounding))}</span>", unsafe_allow_html=True)

    st.markdown(f"**All two-digit combinations including doubles ({len(all_two_digit_combinations)} results):**")
    st.markdown(f"<span style='color:green;'>{', '.join(all_two_digit_combinations)}</span>", unsafe_allow_html=True)
