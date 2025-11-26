from full_forecast import generate_forecast
import mysql.connector
from datetime import datetime

def insert_forecast_to_db(predictions, r2=None, mae=None):
    try:
        conn = mysql.connector.connect(
            host="sql105.infinityfree.com",
            user="if0_40224608",
            password="jrmOGJYzoRJ",
            database="if0_40224608_home_cooking_gee"
        )
        cursor = conn.cursor()

        dishes = predictions.get("dishes", [])
        ingredients = predictions.get("ingredients", [])

        for row in dishes:
            cursor.execute("""
                INSERT INTO weekly_forecast 
                (DishName, ForecastQty, Week, Year, R2Score, MAE, GeneratedAt)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                str(row['DishName']),
                int(row['ForecastQty']),
                int(row['Week']),
                int(row['Year']),
                float(row['R2']) if row.get('R2') is not None else None,
                float(row['MAE']) if row.get('MAE') is not None else None,
                datetime.now()
            ))

        for ing in ingredients:
            cursor.execute("""
                INSERT INTO weekly_ingredients 
                (Ingredient, TotalQty, Unit, Week, Year, GeneratedAt)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                str(ing['Ingredient']),
                float(ing['TotalIngredientQty']),
                str(ing['Unit']),
                int(dishes[0]['Week']) if dishes else None,
                int(dishes[0]['Year']) if dishes else None,
                datetime.now()
            ))

        conn.commit()
        cursor.close()
        conn.close()
        return True

    except Exception as e:
        print(f"❌ Error inserting forecast: {e}")
        return False

if __name__ == "__main__":
    results = generate_forecast()
    if isinstance(results, dict) and "dishes" in results:
        success = insert_forecast_to_db(results)
        if success:
            print("✅ Forecast successfully saved to weekly_forecast and weekly_ingredients tables.")
        else:
            print("❌ Forecast saving failed.")
    else:
        print("❌ Forecast generation failed:", results.get("error", "Unknown error"))
