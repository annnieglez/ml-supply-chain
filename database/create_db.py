'''This script creates the database "supply_chain_db" and the tables. 
It uses SQLAlchemy for database connection and query execution.'''

# Importing the necessary libraries
from sql_connection import get_connection # SQL connection engine
from sqlalchemy import text # SQL query language

def create_database():
    '''This function creates a connection to the MySQL server and
    creates the database "supply_chain_db" if it doesn't exist. It  returns
    the connection object if successful, otherwise it returns None.'''

    # Connect to the MySQL server
    connection = get_connection()

    # Check if the connection was successful
    if connection:

        # Show available databases
        databases = connection.execute(text("SHOW DATABASES"))
        db_list = [db[0] for db in databases.fetchall()]

        # Create the database "supply_chain_db" if it doesn't exist
        if "supply_chain_db" in db_list:
            print("✅ Database 'supply_chain_db' already exists.")
            connection.execute(text("USE supply_chain_db"))
            print("✅ The database supply_chain_db have been selected.")
        else:
            connection.execute(text("CREATE DATABASE supply_chain_db"))
            print("✅ Database 'supply_chain_db' created successfully!")
            connection.execute(text("USE supply_chain_db"))
            print("✅ The database supply_chain_db have been selected.")
        return connection
    else:
        return None

def create_tables(connection):
    '''This function creates the tables by executing the schema.sql script. 
    It receives the connection object as a parameter.'''

    # Execute the schema.sql script to create tables
    with open("schema.sql", "r") as f:
        sql_script = f.read()
        for statement in sql_script.split(";"):  # Execute each statement separately
            if statement.strip(): 
                connection.execute(text(statement))

    connection.close()
    print("✅ Tables created successfully!")

if __name__ == "__main__":  
    '''This script creates the database "supply_chain_db" and the tables.'''

    # Create the database and tables
    connection = create_database()

    if connection != None:
        create_tables(connection)
