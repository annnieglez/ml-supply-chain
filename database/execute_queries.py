'''This script contains functions to execute queries on the database supply_chain_db.'''	

# Importing the necessary libraries
from sql_connection import get_connection # SQL connection engine
from sqlalchemy import text # SQL query language
import pandas as pd # data manipulation

def execute_query(connection, query):
    '''This function executes a query and returns the result if successful,
    otherwise it returns None. It receives the connection object and the query as parameters.'''

    try:
        # Select the database
        connection.execute(text("USE supply_chain_db"))
        print("✅ The database supply_chain_db have been selected.")

        # Execute the query
        result = pd.read_sql(query, connection)
        print(f"✅ Query '{query}' executed successfully.")
        return result
    
    except Exception as e:
        print(f"❌ Error executing query: {query}.")
        return None

def input_query():
    '''This function receives an input query from the user
    and executes it on the database.'''

    # Open a connection to the database
    connection = get_connection()

    while True:
        query = input("Enter your SQL query or command: ")

        # Execute the query
        result = execute_query(connection, query)
        if result is not None:
            print(result)

        # Ask the user if they want to input another query
        another_query = input("Do you want to input another query? (yes/no): ").strip().lower()
        if another_query != 'yes':
            print("❌ Closing the connection. Goodbye!")
            break

    # Close the connection
    connection.close()

if __name__ == "__main__":
    '''This script executes an input query.'''	
    input_query()
