-- This SQL file defines the database structure.

-- Create Customers Table
CREATE TABLE IF NOT EXISTS customers (
    customer_id BIGINT PRIMARY KEY,
    customer_fname VARCHAR(50),
    customer_lname VARCHAR(50),
    customer_email VARCHAR(100),
    customer_password VARCHAR(255),
    customer_city VARCHAR(50),
    customer_state VARCHAR(50),
    customer_zipcode FLOAT,
    customer_country VARCHAR(50),
    customer_segment ENUM('Consumer', 'Corporate', 'Home Office'),
    customer_street VARCHAR(100)
);

-- Create Orders Table
CREATE TABLE IF NOT EXISTS orders (
    order_id BIGINT PRIMARY KEY,
    order_date TIMESTAMP,
    order_status ENUM('COMPLETE', 'PENDING', 'CLOSED', 'PENDING_PAYMENT', 'CANCELED', 'PROCESSING', 'SUSPECTED_FRAUD', 'PAYMENT_REVIEW', 'ON_HOLD'),
    order_region VARCHAR(50),
    order_city VARCHAR(50),
    order_state VARCHAR(50),
    order_country VARCHAR(50),
    order_zipcode VARCHAR(20),
    order_customer_id BIGINT NOT NULL,
    FOREIGN KEY (order_customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE
);

-- Product Categories Table
CREATE TABLE IF NOT EXISTS product_categories (
    category_id BIGINT PRIMARY KEY,
    category_name VARCHAR(100)
);

-- Create Products Table
CREATE TABLE IF NOT EXISTS products (
    product_card_id BIGINT PRIMARY KEY,
    product_description TEXT,
    product_status TINYINT(1),
    product_image TEXT,
    product_name VARCHAR(255),
    product_price DECIMAL(10, 2) CHECK(product_price >= 0),
    product_category_id BIGINT NOT NULL,
    FOREIGN KEY (product_category_id) REFERENCES product_categories(category_id) ON DELETE CASCADE
);

-- Create Orders Items Table
CREATE TABLE IF NOT EXISTS orders_items (
    order_item_id BIGINT PRIMARY KEY,
    order_item_quantity INT CHECK (order_item_quantity > 0),
    order_item_price DECIMAL(13, 9) CHECK (order_item_price >= 0),
    order_item_discount DECIMAL(13, 9) CHECK (order_item_discount >= 0),
    order_item_discount_rate DECIMAL(13, 9) CHECK (order_item_discount_rate BETWEEN 0 AND 1),
    order_item_total DECIMAL(13, 9) CHECK (order_item_total >= 0),
    order_item_profit_ratio DECIMAL(13, 9),
    sales DECIMAL(13, 9) CHECK (sales >= 0),
    order_profit DECIMAL(13, 9),
    benefit_per_order DECIMAL(13, 9),
    sales_per_customer DECIMAL(13, 9) CHECK (sales_per_customer >= 0),
    order_type ENUM('DEBIT', 'TRANSFER', 'PAYMENT', 'CASH'),
    order_id BIGINT NOT NULL,
    order_card_prod_id BIGINT NOT NULL,
    FOREIGN KEY (order_id) REFERENCES orders(order_id) ON DELETE CASCADE,
    FOREIGN KEY (order_card_prod_id) REFERENCES products(product_card_id) ON DELETE CASCADE
);

-- Create Shipping Table
CREATE TABLE IF NOT EXISTS shipping (
    delivery_id BIGINT PRIMARY KEY,
    delivery_status ENUM('Advance shipping', 'Late delivery', 'Shipping canceled', 'Shipping on time'),
    market_name ENUM('Africa', 'Europe', 'LATAM', 'Pacific Asia', 'USCA'),
    shipping_mode ENUM('Standard Class', 'First Class', 'Second Class', 'Same Day'),
    days_for_shipping_real INT CHECK (days_for_shipping_real >= 0),
    days_for_shipping_scheduled INT CHECK (days_for_shipping_scheduled >= 0),
    shipping_date TIMESTAMP,
    late_delivery_risk TINYINT(1) CHECK (late_delivery_risk IN (0,1)), 
    department_id INT,
    department_name VARCHAR(50),
    latitude DECIMAL(10, 6),
    longitude DECIMAL(10, 6),
    order_item_id BIGINT NOT NULL,
    FOREIGN KEY (order_item_id) REFERENCES orders_items(order_item_id) ON DELETE CASCADE
);
