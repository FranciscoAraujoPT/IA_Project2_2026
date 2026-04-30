import Database from 'better-sqlite3';
import { fileURLToPath } from 'url';
import path from 'path';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const db = new Database('data.db');

export function initDb() {
  // Create tables
  db.exec(`
    CREATE TABLE IF NOT EXISTS customers (
      customer_id INTEGER PRIMARY KEY,
      segment TEXT,
      avg_spend REAL,
      patience_level REAL
    );

    CREATE TABLE IF NOT EXISTS products (
      product_id INTEGER PRIMARY KEY,
      category TEXT,
      price REAL,
      complexity INTEGER
    );

    CREATE TABLE IF NOT EXISTS interactions (
      interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
      customer_id INTEGER,
      product_id INTEGER,
      question_time DATETIME,
      response_time_min REAL,
      question_length INTEGER,
      time_of_day TEXT,
      bought BOOLEAN,
      FOREIGN KEY(customer_id) REFERENCES customers(customer_id),
      FOREIGN KEY(product_id) REFERENCES products(product_id)
    );
  `);

  // Seed data if empty
  const customerCount = db.prepare('SELECT count(*) as count FROM customers').get() as { count: number };
  if (customerCount.count === 0) {
    const insertCustomer = db.prepare('INSERT INTO customers (customer_id, segment, avg_spend, patience_level) VALUES (?, ?, ?, ?)');
    const segments = ['low', 'medium', 'high'];
    for (let i = 1; i <= 100; i++) {
      const seg = segments[Math.floor(Math.random() * 3)];
      const patience = seg === 'high' ? 0.8 + Math.random() * 0.2 : (seg === 'medium' ? 0.4 + Math.random() * 0.4 : 0.1 + Math.random() * 0.3);
      insertCustomer.run(i, seg, 100 + Math.random() * 900, patience);
    }

    const insertProduct = db.prepare('INSERT INTO products (product_id, category, price, complexity) VALUES (?, ?, ?, ?)');
    const categories = ['Electronics', 'Home', 'Fashion', 'Beauty'];
    for (let i = 1; i <= 20; i++) {
      insertProduct.run(i, categories[Math.floor(Math.random() * 4)], 10 + Math.random() * 500, Math.floor(Math.random() * 10) + 1);
    }

    const insertInteraction = db.prepare('INSERT INTO interactions (customer_id, product_id, question_time, response_time_min, question_length, time_of_day, bought) VALUES (?, ?, ?, ?, ?, ?, ?)');
    const times = ['morning', 'afternoon', 'evening'];

    for (let i = 0; i < 1000; i++) {
      const customerId = Math.floor(Math.random() * 100) + 1;
      const productId = Math.floor(Math.random() * 20) + 1;

      const customer = db.prepare('SELECT * FROM customers WHERE customer_id = ?').get(customerId) as any;
      const product = db.prepare('SELECT * FROM products WHERE product_id = ?').get(productId) as any;

      // Response time: 1 to 60 mins
      const responseTime = 1 + Math.random() * 59;

      // Logic for purchase probability base on response time
      // Higher response time = lower prob
      // Higher patience = less affected
      // Higher price = lower prob
      let baseProb = 0.5;

      // Response time impact (negative)
      // If patience is 1.0, impact is responseTime / 60 * 0.5
      // If patience is 0.1, impact is responseTime / 60 * 0.9
      const impactFactor = 1.0 - (customer.patience_level * 0.5);
      const rtImpact = (responseTime / 60) * impactFactor;

      baseProb -= rtImpact;

      // Price impact
      baseProb -= (product.price / 1000);

      // Segment impact
      if (customer.segment === 'high') baseProb += 0.1;

      // final prob
      const prob = Math.max(0.05, Math.min(0.95, baseProb));
      const bought = Math.random() < prob ? 1 : 0;

      insertInteraction.run(
        customerId,
        productId,
        new Date().toISOString(),
        responseTime,
        10 + Math.floor(Math.random() * 100),
        times[Math.floor(Math.random() * 3)],
        bought
      );
    }
  }
}

export function getFlattenedData() {
  return db.prepare(`
    SELECT 
      i.response_time_min, 
      p.price, 
      c.patience_level, 
      c.segment, 
      p.complexity, 
      i.time_of_day, 
      i.bought
    FROM interactions i
    JOIN customers c ON i.customer_id = c.customer_id
    JOIN products p ON i.product_id = p.product_id
  `).all() as any[];
}

export function getRawDb() {
  return db;
}

export default db;
