CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS orders (
    order_id              TEXT PRIMARY KEY,
    customer_id           TEXT NOT NULL,
    order_ts              TIMESTAMP NOT NULL,
    order_hour            SMALLINT NOT NULL,
    pincode               TEXT NOT NULL,
    state                 TEXT NOT NULL,
    city_tier             SMALLINT NOT NULL,
    courier               TEXT NOT NULL,
    category              TEXT NOT NULL,
    payment_mode          TEXT NOT NULL,
    order_value           NUMERIC(10, 2) NOT NULL,
    num_items             SMALLINT NOT NULL,
    discount_pct          NUMERIC(5, 2) NOT NULL,
    is_first_order        BOOLEAN NOT NULL,
    customer_prior_orders INTEGER NOT NULL,
    customer_prior_rto    INTEGER NOT NULL,
    address_text          TEXT NOT NULL,
    address_len           SMALLINT NOT NULL,
    address_has_landmark  BOOLEAN NOT NULL,
    customer_name         TEXT NOT NULL,
    customer_phone        TEXT NOT NULL,
    is_rto                BOOLEAN NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_orders_ts ON orders (order_ts);
CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders (customer_id);
