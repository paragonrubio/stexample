import streamlit as st
import pandas as pd

def load_data(uploaded_file):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        return data
    return None

def main():
    st.title("Movie Search App")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        data = load_data(uploaded_file)
        if data is not None:
            st.write("Data loaded successfully!")

            # Search and filter options
            search_title = st.text_input("Search by title")
            search_genre = st.text_input("Search by genre")
            search_director = st.text_input("Search by director")
            search_cast = st.text_input("Search by cast")
            
            # Filter options
            min_year, max_year = int(data["Year"].min()), int(data["Year"].max())
            selected_year = st.slider("Select year", min_year, max_year, (min_year, max_year))

            min_rating, max_rating = float(data["Rating"].min()), float(data["Rating"].max())
            selected_rating = st.slider("Select rating", min_rating, max_rating, (min_rating, max_rating))

            selected_certificate = st.multiselect("Select certificate", data["Certificate"].unique())

            # Apply search and filters
            filtered_data = data[
                (data["Title"].str.contains(search_title, case=False, na=False)) &
                (data["Genre"].str.contains(search_genre, case=False, na=False)) &
                (data["Director"].str.contains(search_director, case=False, na=False)) &
                (data["Cast"].str.contains(search_cast, case=False, na=False)) &
                (data["Year"].between(*selected_year)) &
                (data["Rating"].between(*selected_rating))
            ]

            if selected_certificate:
                filtered_data = filtered_data[filtered_data["Certificate"].isin(selected_certificate)]

            # Display filtered results
            st.write(f"Total results: {filtered_data.shape[0]}")
            st.dataframe(filtered_data)

            # Display movie details
            if not filtered_data.empty:
                selected_movie = st.selectbox("Select a movie to see details", filtered_data["Title"].unique())

                if selected_movie:
                    movie_details = filtered_data[filtered_data["Title"] == selected_movie].iloc[0]
                    st.image(movie_details["Poster"])
                    st.write(f"**Title**: {movie_details['Title']}")
                    st.write(f"**Year**: {movie_details['Year']}")
                    st.write(f"**Certificate**: {movie_details['Certificate']}")
                    st.write(f"**Duration**: {movie_details['Duration (min)']} minutes")
                    st.write(f"**Genre**: {movie_details['Genre']}")
                    st.write(f"**Rating**: {movie_details['Rating']}")
                    st.write(f"**Metascore**: {movie_details['Metascore']}")
                    st.write(f"**Director**: {movie_details['Director']}")
                    st.write(f"**Cast**: {movie_details['Cast']}")
                    st.write(f"**Votes**: {movie_details['Votes']}")
                    st.write(f"**Description**: {movie_details['Description']}")
                    st.write(f"**Review Count**: {movie_details['Review Count']}")
                    st.write(f"**Review Title**: {movie_details['Review Title']}")
                    st.write(f"**Review**: {movie_details['Review']}")

if __name__ == "__main__":
    main()
