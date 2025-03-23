import streamlit as st  
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from PIL import Image
import sqlite3

# Database file
DATABASE = "database.db"

def init_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            security_question TEXT NOT NULL,
            security_answer TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def create_user(username: str, password: str, security_question: str, security_answer: str):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    try:
        cursor.execute('INSERT INTO users (username, password, security_question, security_answer) VALUES (?, ?, ?, ?)', 
                       (username, password, security_question, security_answer))
        conn.commit()
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()
    return True

def get_user(username: str):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
    user = cursor.fetchone()
    conn.close()
    return user

def update_password(username: str, new_password: str):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('UPDATE users SET password = ? WHERE username = ?', (new_password, username))
    conn.commit()
    conn.close()

def load_main_app():
    if not st.session_state.get('logged_in', False):
        st.write("Please log in to access the app.")
        return
    
    st.header("Customer Churn Prediction")
    st.write("Welcome to the Customer Churn Prediction App.")
    
    df = pd.read_csv("customer.csv")
    st.header("Our Dataset")
    
    st.sidebar.header("Filters")
    gender_filter = st.sidebar.multiselect("Gender", options=df['Gender'].unique(), default=df['Gender'].unique())
    age_filter = st.sidebar.slider("Age", min_value=int(df['Age'].min()), max_value=int(df['Age'].max()), 
                                   value=(int(df['Age'].min()), int(df['Age'].max())))
    
    filtered_df = df[df['Gender'].isin(gender_filter) & (df['Age'] >= age_filter[0]) & (df['Age'] <= age_filter[1])]
    st.write(filtered_df)
    
    st.header("After Preprocessing")
    filtered_df['Gender'] = filtered_df['Gender'].map({'Male': 0, 'Female': 1})
    filtered_df['HasCrCard'] = filtered_df['HasCrCard'].map({'Yes': 1, 'No': 0})
    filtered_df['IsActiveMember'] = filtered_df['IsActiveMember'].map({'Yes': 1, 'No': 0})
    st.write(filtered_df)
    
    X = filtered_df[['Gender', 'Age', 'Tenure', 'AccountBalance', 'ProductsNumber', 'IsActiveMember', 'HasCrCard', 'EstimatedSalary']]
    y = filtered_df['Exited']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    st.sidebar.header("Customer Data Input")
    gender = st.sidebar.selectbox("Gender", options=["Male", "Female"])
    age = st.sidebar.number_input("Age", min_value=18, max_value=100)
    tenure = st.sidebar.number_input("Tenure (years)", min_value=0, max_value=100, value=5)
    balance = st.sidebar.number_input("Account Balance", min_value=0, max_value=100000, value=5000)
    product_number = st.sidebar.number_input("Number of Products", min_value=0, max_value=10, value=1)
    has_crcard = st.sidebar.selectbox("Has Credit Card", options=["Yes", "No"])
    estimated_salary = st.sidebar.number_input("Estimated Salary", min_value=0, max_value=1000000, value=5000)
    is_active_member = st.sidebar.selectbox("Is Active Member", options=["Yes", "No"])
    
    gender = 0 if gender == "Male" else 1
    has_crcard = 1 if has_crcard == "Yes" else 0
    is_active_member = 1 if is_active_member == "Yes" else 0
    
    new_data = [[gender, age, tenure, balance, product_number, is_active_member, has_crcard, estimated_salary]]
    prediction = model.predict(new_data)
    prediction_text = "Customer will churn" if prediction[0] == 1 else "Customer will not churn"
    st.sidebar.write(f"Prediction: **{prediction_text}**")
    
    st.subheader("Churn Distribution")
    churn_counts = df["Exited"].value_counts()
    fig, ax = plt.subplots()
    ax.bar(churn_counts.index, churn_counts.values, color=['green', 'red'])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Not Churned', 'Churned'])
    ax.set_ylabel("Number of customers")
    st.pyplot(fig)

def main():
    st.title("Customer Churn Prediction")
    logo = Image.open("customer.png.jpg")
    st.sidebar.image(logo, width=200)
    
    init_db()
    
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    st.sidebar.title("Authentication")
    choice = st.sidebar.radio("Login / Signup / Forgot Password", ["Login", "Signup", "Forgot Password"])
    
    if choice == "Signup":
        username = st.text_input("Username")
        password = st.text_input("Password", type='password')
        security_question = st.text_input("Security Question")
        security_answer = st.text_input("Answer to Security Question", type='password')
        if st.button("Signup"):
            if create_user(username, password, security_question, security_answer):
                st.success("Account created successfully!")
            else:
                st.error("Username already exists.")
    elif choice == "Login":
        username = st.text_input("Username")
        password = st.text_input("Password", type='password')
        if st.button("Login"):
            user = get_user(username)
            if user and user[2] == password:
                st.session_state.logged_in = True
                st.success("Logged in successfully!")
                load_main_app()
            else:
                st.error("Invalid username or password.")
    
if __name__ == "__main__":
    main()
