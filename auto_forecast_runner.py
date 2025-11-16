from full_forecast import generate_forecast
import mysql.connector
from datetime import datetime

# 🔮 Run Forecast
forecast_results = generate_forecast()

# 🧠 Connect to Localhost MySQL
conn = mysql.connector.connect(
    host="sql105.infinityfree.com",
    user="if0_40224608",
    password="jrmOGJYzoRJ",  # Leave blank if no password set in XAMPP
    database="if0_40224608_home_cooking_gee"
)

cursor = conn.cursor()

# 🧾 Insert Forecast Results
for row in forecast_results:
    cursor.execute("""
        INSERT INTO weekly_forecast (DishName, ForecastQty, Week, Year, R2Score, MAE, GeneratedAt)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (
        str(row['DishName']),
        int(row['ForecastQty']),
        int(row['Week']),
        int(row['Year']),
        float(row['R2']),
        float(row['MAE']),
        datetime.now()
    ))

conn.commit()
conn.close()


print("✅ Forecast successfully saved to weekly_forecast table.")
