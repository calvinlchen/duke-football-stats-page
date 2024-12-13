import streamlit as st
import pandas as pd
import numpy as np
import math
from pathlib import Path
from attendance import Attendance

# Title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Duke Football Attendance Predictions',
    page_icon=':football:', # This is an emoji shortcode. Could be a URL too.
)

# -----------------------------------------------------------------------------
# Declare some useful functions.

@st.cache_data
def get_gdp_data():
    """Grab GDP data from a CSV file.

    This uses caching to avoid having to read the file every time. If we were
    reading from an HTTP endpoint instead of a file, it's a good idea to set
    a maximum age to the cache with the TTL argument: @st.cache_data(ttl='1d')
    """

    # Instead of a CSV on disk, you could read from an HTTP endpoint here too.
    DATA_FILENAME = Path(__file__).parent/'data/gdp_data.csv'
    raw_gdp_df = pd.read_csv(DATA_FILENAME)

    MIN_YEAR = 1960
    MAX_YEAR = 2022

    # The data above has columns like:
    # - Country Name
    # - Country Code
    # - [Stuff I don't care about]
    # - GDP for 1960
    # - GDP for 1961
    # - GDP for 1962
    # - ...
    # - GDP for 2022
    #
    # ...but I want this instead:
    # - Country Name
    # - Country Code
    # - Year
    # - GDP
    #
    # So let's pivot all those year-columns into two: Year and GDP
    gdp_df = raw_gdp_df.melt(
        ['Country Code'],
        [str(x) for x in range(MIN_YEAR, MAX_YEAR + 1)],
        'Year',
        'GDP',
    )

    # Convert years from string to integers
    gdp_df['Year'] = pd.to_numeric(gdp_df['Year'])

    return gdp_df

gdp_df = get_gdp_data()

# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.
'''
# :earth_americas: GDP dashboard

Browse GDP data from the [World Bank Open Data](https://data.worldbank.org/) website. As you'll
notice, the data only goes to 2022 right now, and datapoints for certain years are often missing.
But it's otherwise a great (and did I mention _free_?) source of data.
'''

# Add some spacing
''
''

df = pd.read_csv(Path(__file__).parent/'data/DukeAttendanceV10.csv')
# After loading and filtering your DataFrame (let's say it's called df),
# group by 'OppName' and compute the mean attendance.
df = df[df['Site'] == 'Home']
df = df[df['OppName'].isin(Attendance.get_2025_home_opponents(include_noncon=False))]
avg_attendance = df.groupby('OppName')['AttNum'].mean().sort_values(ascending=False)

# Convert this Series to a DataFrame for plotting convenience
avg_attendance_df = avg_attendance.reset_index().rename(columns={'AttNum': 'Average Attendance %'})

st.header("Average Wallace Wade Attendance")
st.write("for each opponent, since 2001 (excluding 2020)")
st.bar_chart(avg_attendance_df.set_index('OppName'))

''
''
df = pd.read_csv(Path(__file__).parent/'data/DukeAttendanceV10.csv')
# After loading and filtering your DataFrame (let's say it's called df),
# group by 'OppName' and compute the mean attendance.
df = df[df['Site'] == 'Home']
df = df[df['Bowl_PrevYear'] == 1]
df = df[df['OppName'].isin(Attendance.get_2025_home_opponents(include_noncon=False))]
avg_attendance = df.groupby('OppName')['AttNum'].mean().sort_values(ascending=False)

# Convert this Series to a DataFrame for plotting convenience
avg_attendance_df = avg_attendance.reset_index().rename(columns={'AttNum': 'Average Attendance %'})

st.header("Average Wallace Wade Attendance")
st.write("for each opponent, since 2001 (excluding 2020)")
st.bar_chart(avg_attendance_df.set_index('OppName'))

''
''

min_value = gdp_df['Year'].min()
max_value = gdp_df['Year'].max()

from_year, to_year = st.slider(
    'Which years are you interested in?',
    min_value=min_value,
    max_value=max_value,
    value=[min_value, max_value])

countries = gdp_df['Country Code'].unique()

if not len(countries):
    st.warning("Select at least one country")

selected_countries = st.multiselect(
    'Which countries would you like to view?',
    countries,
    ['DEU', 'FRA', 'GBR', 'BRA', 'MEX', 'JPN'])

''
''
''

# Filter the data
filtered_gdp_df = gdp_df[
    (gdp_df['Country Code'].isin(selected_countries))
    & (gdp_df['Year'] <= to_year)
    & (from_year <= gdp_df['Year'])
]

st.header('GDP over time', divider='gray')

''

st.line_chart(
    filtered_gdp_df,
    x='Year',
    y='GDP',
    color='Country Code',
)

''
''


first_year = gdp_df[gdp_df['Year'] == from_year]
last_year = gdp_df[gdp_df['Year'] == to_year]

st.header(f'GDP in {to_year}', divider='gray')

''

cols = st.columns(5)

for i, country in enumerate(selected_countries):
    col = cols[i % len(cols)]

    with col:
        first_gdp = first_year[first_year['Country Code'] == country]['GDP'].iat[0] / 1000000000
        last_gdp = last_year[last_year['Country Code'] == country]['GDP'].iat[0] / 1000000000

        if math.isnan(first_gdp):
            growth = 'n/a'
            delta_color = 'off'
        else:
            growth = f'{last_gdp / first_gdp:,.2f}x'
            delta_color = 'normal'

        st.metric(
            label=f'{country} GDP',
            value=f'{last_gdp:,.0f}B',
            delta=growth,
            delta_color=delta_color
        )

# Run the prediction function (which returns a DataFrame)
Attendance.prepare_model()

# df_2025 = Attendance.test_prediction()

# st.title("Predicted Attendance for 2025 Games")

# # Iterate through the rows of the resulting DataFrame
# for i, row in df_2025.iterrows():
#     # row is a pandas Series; access columns like a dictionary
#     st.metric(
#         label=row['OppName'],
#         value=int(row['PredictedAttendance'])
#     )

# Add the new columns to df_2025 if they don't exist
# Initially, set them to False or some default values
if 'OppRankedGametime' not in df_2025.columns:
    df_2025['OppRankedGametime'] = False

if 'Rain' not in df_2025.columns:
    df_2025['Rain'] = False

if 'DukeWins' not in df_2025.columns:
    df_2025['DukeWins'] = False

st.write("Adjust the conditions for each game and then click 'Update Predictions'.")

# Create a form to handle user input
with st.form("update_predictions_form"):
    for i, row in df_2025.iterrows():
        st.write(f"### {row['OppName']}")
        # We'll create three checkboxes per game. We use a unique key for each row.
        opp_ranked = st.checkbox("Opponent Ranked?", value=row['OppRankedGametime'], key=f"OppRanked_{i}")
        rain = st.checkbox("Rain?", value=row['Rain'], key=f"Rain_{i}")
        duke_wins = st.checkbox("Duke Wins?", value=row['DukeWins'], key=f"DukeWins_{i}")

    # A submit button to update predictions
    submitted = st.form_submit_button("Update Predictions")

    if submitted:
        # Update df_2025 with the user inputs
        for i, row in df_2025.iterrows():
            df_2025.at[i, 'OppRankedGametime'] = st.session_state[f"OppRankedGametime_{i}"]
            df_2025.at[i, 'Rain'] = st.session_state[f"Rain_{i}"]
            df_2025.at[i, 'DukeWins'] = st.session_state[f"DukeWins_{i}"]

        # Now we need to re-run the prediction with updated features
        # Make sure your attendance.py model also uses these features in 'features' list.
        # For example, in attendance.py:
        # features = ['OppFPI_PrevYear', 'OppCityDist', 'MaxCapacity',
        #             'ThanksgivingWeekend', 'LaborDayWeekend', 'UNC_Game', 'OppName', 
        #             'OppRanked', 'Rain', 'DukeWins']

        # Run the model's prediction again with updated conditions
        # We need a method that can just take df_2025 and predict again.
        # Let's assume we have a separate method in Attendance class like "Attendance.predict(df)" 
        # or just replicate the code here.
        
        # If we have a method in Attendance that can predict given a DF:
        updated_predictions = Attendance.predict(df_2025)  # You need to implement this in attendance.py

        df_2025['PredictedAttendance'] = updated_predictions

        st.success("Predictions updated!")
        # Display results
        for i, row in df_2025.iterrows():
            st.metric(
                label=row['OppName'],
                value=int(row['PredictedAttendance'])
            )