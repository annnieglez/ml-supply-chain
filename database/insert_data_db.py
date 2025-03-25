
""" This script is responsible for inserting the data into the database tables. """

# Importing the necessary libraries
import mysql.connector
from getpass import getpass # get password without showing it

def insert_data():
    """    Establishes a connection to the database, inserts predefined 
    sample data into the tables."""

    # Connect to the MySQL server
    #connection = get_connection()

    try: 
        password = getpass("Please, insert your password: ")

        conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password=password,
        database="supply_chain_db",
        allow_local_infile=True
        )

        connection = conn.cursor()

        if connection:
            print("✅ Connection successfully opened.")
            print("✅ The database supply_chain_db have been selected.")
    
            # Execute the insert_data.sql script to create tables
            with open("insert_data.sql", "r") as f:
                sql_script = f.read()
                
                for statement in sql_script.split(";"):  # Execute each statement separately
                    if statement.strip(): 
                        connection.execute(statement)
                        conn.commit()

            connection.close()
            conn.close()
            print("✅ Data inserted successfully!")

        else:
            print("❌ Failed to open connection. Try again!")
            return
    
    except Exception as e:
        print(f"❌ Failed: {e}")

if __name__ == "__main__":
    insert_data()

