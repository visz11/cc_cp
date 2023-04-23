from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

# Load the flight data
flight_data = pd.read_csv('flight_data1.csv')

# Scale the flight ratings
scaler = MinMaxScaler()
flight_data['Rating'] = scaler.fit_transform(flight_data[['Rating']])

# Fit the KNN model
knn_model = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='cosine')
knn_model.fit(flight_data[['Rating']])

# Define the Flask app
app = Flask(__name__)

# Define the route for the home page
@app.route('/')
def home():
    return render_template('home.html')

# Define the route for the recommendation page
@app.route('/recommend', methods=['POST'])
def recommend():
    # Get the user's input
    global flight_data

    user_input = request.form['user_input']
    user_rating = request.form['user_rating']
    
    # Add the user's input to the flight data
    new_row = {'Flight': 'User Input', 'Description': user_input, 'Rating': user_rating}
    # flight_data = flight_data.append(new_row, ignore_index=True)
    new_row_df = pd.DataFrame([new_row])
    flight_data= pd.concat([flight_data, new_row_df], ignore_index=True)

    # Scale the user's rating
    user_rating_scaled = scaler.transform([[user_rating]])[0][0]
    
    # Fit the KNN model on the updated flight data
    knn_model.fit(flight_data[['Rating']])
    
    # Find the nearest neighbors
    distances, indices = knn_model.kneighbors([[user_rating_scaled]])
    
    # Get the recommended flights
    recommended_flights = flight_data.iloc[indices[0][1:]]
    
    # Remove the user's input from the flight data
    flight_data.drop(flight_data.tail(1).index, inplace=True)
    
    # Render the recommendation page
    return render_template('recommend.html', recommended_flights=recommended_flights)

if __name__ == '__main__':
    app.run(debug=False)