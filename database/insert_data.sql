-- This SQL file data into the 'supply_chain_db' database.

-- Insert data into Customers Table
LOAD DATA LOCAL INFILE '.\\data\\customers_db.csv' 
INTO TABLE customers 
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"' 
LINES TERMINATED BY '\n'
IGNORE 1 ROWS; 

-- Insert data into Orders Table
SET @@SESSION.time_zone = '+00:00';
LOAD DATA LOCAL INFILE '.\\data\\orders_db.csv' 
INTO TABLE orders 
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"' 
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

-- Insert data into Product Categories Table
LOAD DATA LOCAL INFILE '.\\data\\product_category_db.csv' 
INTO TABLE product_categories 
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"' 
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

-- Insert data into Products Table
LOAD DATA LOCAL INFILE '.\\data\\products_db.csv' 
INTO TABLE products 
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"' 
LINES TERMINATED BY '\n'
IGNORE 1 ROWS; 

-- Insert data into Orders Items Table
LOAD DATA LOCAL INFILE '.\\data\\order_items_db.csv' 
INTO TABLE orders_items 
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"' 
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

-- Insert data into Shipping Table
LOAD DATA LOCAL INFILE '.\\data\\shipping_db.csv' 
INTO TABLE shipping 
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"' 
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;