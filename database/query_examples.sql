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

-- Find orders with negative benefit and flags as suspected fraud in a specific time period
SELECT *  
FROM orders o
JOIN orders_items oi ON o.order_id = oi.order_id
JOIN shipping s ON oi.order_id = s.order_item_id
WHERE o.order_date BETWEEN '2017-09-29' AND '2017-10-04'
AND benefit_per_order < 0
AND o.order_status = 'SUSPECTED_FRAUD'
ORDER BY oi.benefit_per_order ASC;