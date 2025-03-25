USE supply_chain_db;

-- Get all products categories
SELECT * FROM product_categories;

-- Get total orders per product
SELECT p.product_name, COUNT(o.order_id) AS total
FROM orders o
JOIN orders_items oi ON o.order_id = oi.order_id
JOIN products p ON oi.order_card_prod_id = p.product_card_id
GROUP BY p.product_name
ORDER BY total DESC;

-- Find the most expensive product
SELECT p.product_name, p.product_price FROM products p ORDER BY p.product_price DESC LIMIT 1;