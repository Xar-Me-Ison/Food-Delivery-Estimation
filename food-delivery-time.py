# Διαχείριση Μεγάλου Όγκου Δεδομένων στο Διαδίκτυο
# Μέλη Ομάδας:
# Βασίλειος Κατωτομιχελάκης (Π2020132)
# Χαράλαμπος Μακρυλάκης (Π2019214)

import os # Για την διαχείριση εντολών εντός του λειτουργικού συστήματος
import codecs # Για το encoding κατά την εμφάνιση του αρχείου README.md
import streamlit as st # Για την δημιουργία της ιστοσελίδας
import pyspark as ps # Για την ανάλυση του μεγάλου όγκου δεδομένων
import math as mt # Για την χρήση γενικών μαθηματικών συναρτήσεων
import numpy as np # Για την χρήση Γραμμικής Άλγεβρας
import pandas as pd # Για την επεξεργασία δεδομένων και το CSV Ι/Ο
import matplotlib.pyplot as plt # Για την δημιουργία διαγραμμάτων plotting chart
import seaborn as sns # Για την οπτικοποίηση συγκεκριμένων διαγραμμάτων
from sklearn.linear_model import LinearRegression # Για το μοντέλο εκμάθησης Linear Regression
from sklearn.ensemble import ExtraTreesRegressor # Για το μοντέλο εκμάθησης Extra Trees Forest 
from sklearn.ensemble import RandomForestRegressor # Για το μοντέλο εκμάθησης Random Forest Regression
from sklearn.ensemble import GradientBoostingRegressor # Για το μοντέλο εκμάθησης Gradient Boosting Regression
from sklearn.neighbors import KNeighborsRegressor # Για το μοντέλο εκμάθησης K-Neighbors Regression
from sklearn.tree import DecisionTreeRegressor # Για το μοντέλο εκμάθησης Decision Tree Regression
from sklearn.model_selection import train_test_split # Για τον διαχωρισμό σε training και test sets
from sklearn.metrics import mean_squared_error # Για τον υπολογισμό του MSE
from sklearn.metrics import mean_absolute_error # Για τον υπολογισμό του MAE
from sklearn.metrics import r2_score # Για τον υπολογισμό του R-squared score



# Δημιουργία (global) dataframe για την διαχείριση δεδομένων του dataset σε όλο το φάσμα του προγράμματος
data_frame = pd.read_csv('food-delivery-time.csv')

# Συνάρτηση Haversine για τον υπολογισμό αποστάσεων μεταξύ γεωγραφικού πλάτους και γεωγραφικού μήκους.
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Το ρ της γης είναι 6371 km.

    # Μετατροπή από γεωγραφικού πλάτους και μήκους σε radians.
    lat1 = mt.radians(lat1)
    lon1 = mt.radians(lon1)
    lat2 = mt.radians(lat2)
    lon2 = mt.radians(lon2)

    # Υπολογισμός διαφερός κάθε γεωγραφικής θέσεις
    Δlat = lat2 - lat1
    Δlon = lon2 - lon1

    # Φόρμουλα Haversine
    a = mt.sin(Δlat/2)**2 + mt.cos(lat1) * mt.cos(lat2) * mt.sin(Δlon/2)**2
    c = 2 * mt.atan2(mt.sqrt(a), mt.sqrt(1-a))
    d = R * c

    return d


# Δημιουργία νέας στήλης ονόματι 'Distance' χρησιμοποιώντας την φόρμουλα Haversine
data_frame['Distance'] = data_frame.apply(lambda row: haversine(row['Restaurant_latitude'], row['Restaurant_longitude'],
                                                   row['Delivery_location_latitude'], row['Delivery_location_longitude']), axis=1)

# Επιλογή των 'selected_features' και 'target_variable' για τα μοντέλα εκμάθησης
selected_features = ['Restaurant_latitude', 'Restaurant_longitude', 'Delivery_location_latitude', 'Delivery_location_longitude', 'Distance', 'Delivery_person_Age', 'Type_of_vehicle']
target_variable = 'Time_taken(min)'

# Προετοιμασία των feature set και target variable
X = data_frame[selected_features]
y = data_frame[target_variable]

# Μετατροπή κατηγοριοποιημένης μεταβλητής 'Type_of_vehicle' σε numerical με την χρήση one-hot encoding
X = pd.get_dummies(X, columns=['Type_of_vehicle'], drop_first=True)

# Διαχωρισμός των δεδομένων σε training και testing sets
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)


# Συνάρτηση για την συγγραφή παραγράφων με πιο εύκολο και ευανάγνωστο τρόπο
def streamlit_write(header='', text='', markdown=False):
    valid_headers = {
        'h1': '#',
        'h2': '##',
        'h3': '###',
        'h4': '####',
        'h5': '#####',
        'h6': '######'
    }

    if markdown == True:
        st.markdown("---")

    if header in valid_headers:
        text = f"{valid_headers[header]} {text}"
        st.write(text)
    else:
        st.write(text)


# Σύναρτηση εκκίνησης και την γενική λογική της ιστοσελίδας Streamlit 
def streamlit_init():
    # Για την εμφάνιση της ιστοσελίδας
    st.set_page_config(page_title='Διαχείριση Μεγάλου Όγκου Δεδομένων στο Διαδίκτυο - Π2019214, Π2020132')
    st.title("Food Delivery Time Estimation")
    
    # Δημιουργία των υποσελίδων εντός της ιστοσελίδα μας
    tabs = ["Application", "Data Analysis", "README"]
    active_tab = st.selectbox("Select Tab", tabs)

    if active_tab == "Application":
        application()
    elif active_tab == "Data Analysis":
        data_analysis()
    elif active_tab == "README":
        read_me()

# Συνάρτηση για το κομμάτι/tab Application της ιστοσελίδας μας
def application():
    # Εμφάνιση header για το συγκεκριμένο tab
    st.header("Application")

    # ---- HEADER: Μικρή επεξήγηση για την λειτουργία του προγράμματος  ----
    streamlit_write(header='h5', text="Σκοπός του προγράμματος είναι ο υπολογισμός της εκτιμώμενης παράδοσης μιας παραγγελίας με μεγάλο ποσοστό ευστοχίας.")
    st.markdown('---')
    # ---- HEADER: Μικρή επεξήγηση για την λειτουργία του προγράμματος  ----
    
    # ---- SECTION: Food Delivery Time Calculator ----
    training_algorithm = st.selectbox("Training algorithm 🤖", ["Random Forest Regression",  "Gradient Boosting Regression", "Linear Regression", "Decision Tree Regression", "Extra Trees Forest", "K-Neighbors Regression"])

    # Split the app layout into two columns
    col1, col2 = st.columns(2)

    # Inputs in the first column
    with col1:
        driver_age = st.number_input("Delivery driver age 📅", min_value=18, max_value=120, value=25)
        
        customer_location_latitude = st.number_input("Customer location latitude📍",  min_value=-90, max_value=90)
        customer_location_longitude = st.number_input("Customer location longtitude📍",  min_value=-180, max_value=180) 


    # Inputs in the second column
    with col2:
        driver_vehicle = st.selectbox("Type of vehicle 🏍️", ["Motorcycle", "Scooter", "Electric scooter", "Bicycle"])

        restaurant_location_latitude = st.number_input("Restaurant location latitude🗺️",  min_value=-90, max_value=90)
        restaurant_location_longitude = st.number_input("Restaurant location longtitude🗺️",  min_value=-180, max_value=180)

    more_info = st.checkbox("Display more information about an algorithm's accuracy such as MSE, RMSE, MAE and R².")

    calculate_button = st.button("Calculate")

    if calculate_button:
        # Υπολογισμός απόστασης με την χρήση της συνάρτησης Haversine
        distance = haversine(restaurant_location_latitude, restaurant_location_longitude,
                            customer_location_latitude, customer_location_longitude)

        # Μετατροπή των οχημάτων με one-hot encoded format για τα μοντέλα εκμάθησης
        vehicle_df = pd.DataFrame(columns=['Type_of_vehicle'])
        vehicle_df.loc[0] = driver_vehicle
        vehicle_df = pd.get_dummies(vehicle_df, columns=['Type_of_vehicle'], drop_first=True)


        # Προετοιμασία του input ως DataFrame με ονοματισμένες στήλες για την πρόβλεψη χρόνου
        # Prepare the input for prediction as a DataFrame with named columns
        user_input_for_prediction = pd.DataFrame({
            'Restaurant_latitude': [restaurant_location_latitude],
            'Restaurant_longitude': [restaurant_location_longitude],
            'Delivery_location_latitude': [customer_location_latitude],
            'Delivery_location_longitude': [customer_location_longitude],
            'Distance': [distance],
            'Delivery_person_Age': [driver_age]
        })

        # 
        #Συγχώνευση δεδομένων οχήματος με την είσοδο για πρόβλεψη
        user_input_for_prediction = pd.concat([user_input_for_prediction, vehicle_df], axis=1)
        
        # Βεβαίωση ότι οι στήλες ταιριάζουν με αυτές που χρησιμοποιούνται κατά τη διάρκεια της εκμάθησης 
        missing_cols = set(X_train.columns) - set(user_input_for_prediction.columns)
        for col in missing_cols:
            user_input_for_prediction[col] = 0
        
        # Τακτοποίηση των στηλών με την ίδια σειρά όπως κατά τη διάρκεια της εκπαίδευσης
        user_input_for_prediction = user_input_for_prediction[X_train.columns]
        
        # Εκπαίδευση μοντέλου εκμάθησης βάση τα επιλεγμένα χαρακτηριστικά
        if training_algorithm == "Random Forest Regression":
            model = RandomForestRegressor()
        elif training_algorithm == "Gradient Boosting Regression":
            model = GradientBoostingRegressor()
        elif training_algorithm == "Linear Regression":
            model = LinearRegression()
        elif training_algorithm == "Decision Tree Regression":
            model = DecisionTreeRegressor()
        elif training_algorithm == "Extra Trees Forest":
            model = ExtraTreesRegressor()
        else:
            model = KNeighborsRegressor()

        model.fit(X_train, y_train)
        
       # Πρόβλεψη χρόνου παράδοσης
        estimated_time = model.predict(user_input_for_prediction)[0]

        # Υπολογισμός MSE, MAE και R-squared αντίστοιχα
        y_true = y_train
        y_pred = model.predict(X_train)

        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        info_text = f"Mean Squared Error (MSE): {mse:.4f}\n\n"\
                    f"Root Mean Squared Error (RMSE): {rmse:.4f}\n\n"\
                    f"Mean Absolute Error (MAE): {mae:.4f}\n\n"\
                    f"R-squared (R²) Score: {r2:.4f}\n\n"\
                    
        st.success(f"The estimated delivery time is approximately {estimated_time:.1f} minutes.")
        if more_info:
            st.info(info_text)
    # ---- SECTION: Food Delivery Time Calculator ----

# Συνάρτηση για το κομμάτι/tab Data Analysis της ιστοσελίδας μας
def data_analysis():
    # Εμφάνιση header για το συγκεκριμένο tab
    st.header("Data Analysis")
    
    # ---- HEADER: Μικρή εισαγωγή για την σελίδα της εξερευνητικής ανάλυσης δεδομένων ----
    streamlit_write(header='h5', text="Στην εξερευνητική ανάλυση δεδομένων αυτή, θα γίνει επεξεργασία, επισκόπηση καθώς και καθαρισμός των δεδομένων για την βελτίωση της ποιότητάς τους και την αποτελεσματικότητα της ανάλυσης.")
    # ---- HEADER: Μικρή εισαγωγή για την σελίδα της εξερευνητικής ανάλυσης δεδομένων ----


    # ---- SECTION 1: Εμφάνιση του dataframe ----
    streamlit_write(header='h4', text="1. Εμφάνιση του dataframe", markdown=True)
    # Εμφάνιση των πρώτων 5 γραμμών του dataframe
    streamlit_write(header='h5', text="Εμφάνιση των πρώτων 5 γραμμών του dataframe, για να βεβαιωθούμε ότι εισαγωγή των δεδομένων εκτελέστηκε επιτυχώς.")
    st.code("data_frame.head()")
    st.write(data_frame.head())

    # Αντίστοιχα, εμφάνιση των τελευταίων 5 γραμμών του dataframe
    streamlit_write(header='h5', text="Αντίστοιχα για τις τελευταίες 5 γραμμές.")
    st.code("data_frame.tail()")
    st.write(data_frame.tail())
    streamlit_write(header='h5', text="Όπως βλέπουμε σχεδόν 46.000 εγγραφές, συγκριτικά μικρό με άλλα datasets αλλά επαρκές για την περίπτωση μας.")
    # ---- SECTION 1: Εμφάνιση του dataframe ----


    # ---- SECTION 2: Καθαρισμός των αρχικών δεδομένων ----
    streamlit_write(header='h4', text="2. Καθαρισμός των αρχικών δεδομένων", markdown=True)
    # Καθαρισμός και εμφάνιση των στηλών των δεδομένων
    streamlit_write(header='h5', text=" Καθαρισμός των στηλών των δεδομένων με την εντολή:")
    st.code("category_column = data_frame.select_dtypes(include='object').columns")
    category_column = data_frame.select_dtypes(include='object').columns
    st.write(category_column)

    # Εμφάνιση των numerical ονομάτων των στηλών
    numeral_columns = data_frame.drop(category_column, axis=1).columns
    streamlit_write(header='h5', text=" Εμφάνιση των numerical ονομάτων των στηλών.")
    st.code("numeral_columns = data_frame.drop(category_column, axis=1).columns")
    st.write(numeral_columns)

    # Έλεγχος για τον καθαρισμό δεδομένων, ελέγχοντας αν υπάρχουν κενά πεδία
    streamlit_write(header='h5', text=" Έλεγχος για τον καθαρισμό δεδομένων, ελέγχοντας αν υπάρχουν κενά πεδία.")
    streamlit_write(header='h5', text=" Ευτυχώς, τα δεδομένα μας είναι ολοκληρομένα, αλλά συνήθως ο καθαρισμός είναι απαραίτητο κομμάτι της Διαχείρηση Μεγάλου Όγκου Δεδομένων.")
    st.code("data_frame.isnull().sum()")
    st.write(data_frame.isnull().sum())

    # Επιπλέον έλεγχος για τον καθαρισμό δεδομένων, όπου ελέγχουμε αν υπάρχουν διπλότυπες στύλες μέσα στο dataset μας
    streamlit_write(header='h5', text=" Επιπλέον έλεγχος για τον καθαρισμό δεδομένων, όπου ελέγχουμε αν υπάρχουν διπλότυπες στύλες μέσα στο dataset μας")
    st.code("data_frame.duplicated().sum()")
    st.write("Διπλότυπες τιμές: ", data_frame.duplicated().sum())

    # Εκτύπωση των ακρών των δεδομένων μας για έλεγχο
    streamlit_write(header='h5', text="Για άλλη μια φορά, δεν ύπαρχουν διπλότυπα στα δεδομένα μας, το οποίο σημαίνει ότι είναι ολοκληρομένα.\n")
    streamlit_write(header='h5', text="Ας εμφανίσουμε, για τελευταία φορά, τις άκρες των δεδομένων μας για έλεγχο.")
    st.code('''    for col in category_column:
        st.write(data_frame[col].value_counts())
        st.write()''')
    
    # Διαχωρισμός της διάταξης της εφαρμογής σε πολλές στήλες με βάση τον αριθμό του 'category_column'
    num_columns = len(category_column) // 2 + len(category_column) % 2
    cols = st.columns(num_columns)
    # Display value counts for each category column in separate columns
    for i in range(0, len(category_column), 2):
        with cols[i // 2]:
            st.write(f"###### Value Counts for {category_column[i]}")
            st.write(data_frame[category_column[i]].value_counts())

            if i + 1 < len(category_column):
                st.write(f"###### Value Counts for {category_column[i + 1]}")
                st.write(data_frame[category_column[i + 1]].value_counts())

    streamlit_write(header='h5', text="Από ότι βλέπουμε, όχι μόνο δεν έχουμε πρόβλημα με τα δεδομένα μας, αλλά έχουμε μια ίση σχεδόν κατανομή των πεδίων. Το οποίο μας γλυτώνει την δουλειά της κανονικοποίησης. Ας συνεχίσουμε στο επόμενο βήμα.")

    # Εφαρμογή του Five Number Summary στα δεδομένα μας
    streamlit_write(header='h5', text="Ας κάνουμε το Five Number Summary (minimum, first quartile, median, third quartile, and maximum) και να δούμε τι θας μας βγάλει:")
    st.code("data_frame.describe().transpose()")
    st.write(data_frame.describe().transpose())
    # ---- SECTION 2: Καθαρισμός των αρχικών δεδομένων ----


    # ---- SECTION 3: Οπτικοποίηση δεδομένων ----
    streamlit_write(header='h4', text="3. Οπτικοποίηση δεδομένων", markdown=True)
    streamlit_write(header='h5', text="Σε αυτό το κομμάτι θα κάνουμε χρήση βιβλιοθηκών όπως `matplotlib` και `seaborn` για την οπτικοποίηση των δεδομένων μας.")
    # Οπτική αναπαράστηση του πεδίου 'Type of Order'
    plt.figure(dpi=100)
    plt.title("Ranked types of order")
    sns.countplot(data=data_frame, x='Type_of_order')
    streamlit_write(header='h5', text="Για παράδειγμα την οπτικοποίηση του πεδίου 'Type of Order':")
    st.pyplot(plt)
    streamlit_write(header='h5', text="Όπως παρατηρήσαμε και στο αριθμητικό παράδειγμα, η κατανομή των πεδίων μας είναι ίση, το οποίο μας γλυτώνει την κανονικοποίηση των δεδομένων.")

    # Οπτική αναπαράστηση των τύπων οχημάτων που χρησιμοποιήθηκαν 
    plt.figure(dpi=100)
    plt.title("Type of Vehicles Ranked")
    sns.countplot(data=data_frame,x='Type_of_vehicle')
    streamlit_write(header='h5', text="Ας κάνουμε το ίδιο για των τύπων οχημάτων που χρησιμοποιήθηκαν ('Type_of_vehicle'):")
    st.pyplot(plt)


    # Οπτική αναπαραστάση των πέντε υψηλότερων ID του dataframe μας χρησιμοποιώντας slices 
    ID_dataframe = data_frame['ID'].value_counts().reset_index()[:5]
    plt.figure(dpi=100)
    plt.title("Top 5 IDs")
    sns.barplot(data=ID_dataframe, x='ID', y='count')
    streamlit_write(header='h5', text="Τώρα, ας αναπαραστήσουμε τα 5 ηψηλότερα IDs.")
    st.pyplot(plt)
    
    # Οπτική αναπάρασταση των 5 καλύτερων διανομέων χρησιμοποιώντας slices 
    delivery_person_ID_dataframe = data_frame['Delivery_person_ID'].value_counts().reset_index()[:5]
    plt.figure(dpi=100)
    plt.title("Top 5 Delivery Person IDs")
    sns.barplot(data=delivery_person_ID_dataframe, x='Delivery_person_ID', y='count')
    plt.xticks(rotation=90);
    streamlit_write(header='h5', text="Και στην συνέχεια, ας κάνουμε την οπτική αναπάρασταση των 5 καλύτερων διανομέων, για να δούμε το γενικό count, και την απόδοση τους.")
    st.pyplot(plt)

    # Εμφάνιση heatmap για την εύρεση των πιο σημαντικών μεταβλητών
    plt.figure(dpi=100)
    plt.title('Heatmap Delivery Data')
    sns.heatmap(data_frame[numeral_columns].corr(), annot=True, cmap="YlGnBu")
    streamlit_write(header='h5', text="Κάνοντας το heatmap μπορούμε να διαπυστώσουμε σε ποια πεδία τιμών να εστιάσουμε, και γενικότερα έχουν την μεγαλύτερη συσχέτιση μεταξύ τους.")
    st.pyplot(plt)
    streamlit_write(header='h5', text="Όπως βλέπουμε, τα πεδία που έχουν την σημαντικότερη συσχέτιση είναι αυτά που έχουν να κάνουνε με την τοποθεσία του εστιατορίου/μαγαζιού (`'Restaurant_latitude', 'Restaurant_longitude'`) και αντίστοιχα του πελάτη ή της παραγγελίας (`'Delivery_location_latitude', 'Delivery_location_longitude'`).")
    streamlit_write(header='h5', text="Άρα, στην ερεύνηση και ανάπτυξη της εφαρμογής μας για τον υπολογισμό της αναμενόμενης ώρας για μια παραγγελία, θα πάρουμε αυτά τα πεδία υπόψη μας, καθώς έχουν την μεγαλύτερη βαρύτητα στην πρόβλεψη του χρόνου παράδοσης.")

    # Αναπαράσταση της σχέσης χρόνου κάθε στήλης
    time_taken_corr_dataframe = data_frame[numeral_columns].corr()['Time_taken(min)'].reset_index().sort_values('Time_taken(min)')[:-1]
    plt.figure(dpi=100)
    sns.barplot(data=time_taken_corr_dataframe, x='index', y='Time_taken(min)')
    plt.title('Correlation Of Columns With Time Taken')
    plt.ylabel('Correlation')
    plt.xlabel('Column Names')
    plt.xticks(rotation=90);
    streamlit_write(header='h5', text="Συνεπώς, ας βρούμε την συσχέτιση του χρόνου (time taken), με κάθε πεδίου της στήλης μας.")
    st.pyplot(plt)
    streamlit_write(header='h5', text="Διαγραμματικά μας δείχνετε, ότι όσον αφορά την σχέση χρόνου του πεδίου αξιολόγησης (`'Delivery_person_ratings'`), έχουμε αρνητική συσχέτιση. Δηλαδή, η επιρροή του στο χρόνο είναι ανύπαρκτη.")
    streamlit_write(header='h5', text="Αντιθέτως, η σχέση χρόνο με το πεδίο της ηλικίας του διανομέα (`'Delivery_person_age'`), φαίνεται να έχει την μεγαλύτερη συσχέτιση. Και ως επακόλουθο, κατά τον υπολογισμό και πρόβλεψη του χρόνου διανομής, θα πάρουμε υπόψη την ηλικία του διανομέα.")
    # ---- SECTION 3: Οπτικοποίηση δεδομένων ----

# Συνάρτηση για το κομμάτι/tab README.md της ιστοσελίδας μας
def read_me():
    # Εμφάνιση header για το συγκεκριμένο tab
    st.header("README")

    # ---- HEADER: Μικρή επεξήγηση για τις βιβλιοθήκες που χρησιμοποιήσαμε και τον τρόπο εκτέλεσης της εφαρμογής  ----
    streamlit_write(header='h5', text="Σκοπός του παρόν αρχείου είναι να δείξουμε συνοπτικά τις βιβλιοθήκες που χρησιμοποιήσαμε και τον τρόπο εκτέλεσης της εφαρμογής.")
    st.markdown('---')
    # ---- HEADER: Μικρή επεξήγηση για τις βιβλιοθήκες που χρησιμοποιήσαμε και τον τρόπο εκτέλεσης της εφαρμογής  ----
    readme_path = "README.md"

    if os.path.exists(readme_path):
        with codecs.open(readme_path, "r", encoding="utf-8", errors="ignore") as readme_file:
            readme_content = readme_file.read()
            st.markdown(readme_content)
    else:
        st.write("README.md file not found.")

def main():
    streamlit_init()


if __name__ == "__main__":
    main()