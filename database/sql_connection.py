''' This script contains the function to connect to the MySQL server. '''

# Importing the necessary libraries
from getpass import getpass # get password without showing it
from sqlalchemy import create_engine # SQL connection engine

# Function to connect to the database
def get_connection():
    '''This function creates a connection to the MySQL server and
    returns the connection object if successful, otherwise it returns None.'''

    try:
        # Connect to the MySQL server
        password = getpass("Please, insert your password: ")
        db_url = f"mysql+pymysql://root:{password}@localhost"
        engine = create_engine(db_url, connect_args={"local_infile": 1},
            execution_options={"autocommit": True})
        connection = engine.connect()

        if connection:
            print("✅ Connection successfully opened.")
            return connection
        else:
            print("❌ Failed to open connection. Try again!")
            return None
    
    except Exception as e:
        print(f"❌ Failed to open connection, might be due to wrong password!")

    # Check if the connection was successful
    

    
if __name__ == "__main__":  
    '''This script open the connection.'''

    # Create the database and tables
    connection = get_connection()
