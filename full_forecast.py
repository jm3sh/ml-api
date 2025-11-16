# 📦 1. Import Libraries
import pandas as pd
import numpy as np
import math
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def generate_forecast():
    # 📁 2. Load CSV Files
    orders_df = pd.read_csv('order_data.csv')
    ingredients_df = pd.read_csv('dishingredients.csv')

    # 🧼 3. Clean Column Names
    orders_df.columns = [str(col).strip() for col in orders_df.columns]
    ingredients_df.columns = [str(col).strip() for col in ingredients_df.columns]

    # 🕒 4. Convert OrderDate to datetime
    orders_df['OrderDate'] = pd.to_datetime(orders_df['OrderDate'])

    # 📅 5. Define Holidays
    holiday_dates = pd.to_datetime([
        '2024-01-01', '2024-03-28', '2024-03-29', '2024-04-09', '2024-05-01',
        '2024-06-12', '2024-08-23', '2024-08-26', '2024-11-30', '2024-12-25', '2024-12-30',
        '2025-01-01', '2025-01-29', '2025-04-09', '2025-04-17', '2025-04-18', '2025-04-19',
        '2025-05-01', '2025-06-12', '2025-08-21', '2025-08-25', '2025-10-31', '2025-11-01',
        '2025-11-30', '2025-12-08', '2025-12-24', '2025-12-25', '2025-12-30', '2025-12-31'
    ])

    # 🧮 6. Add Seasonal Features
    orders_df['Year'] = orders_df['OrderDate'].dt.year
    orders_df['Week'] = orders_df['OrderDate'].dt.isocalendar().week
    orders_df['Month'] = orders_df['OrderDate'].dt.month
    orders_df['DayOfWeek'] = orders_df['OrderDate'].dt.dayofweek
    orders_df['IsWeekend'] = orders_df['DayOfWeek'].isin([5, 6]).astype(int)
    orders_df['IsHoliday'] = orders_df['OrderDate'].isin(holiday_dates).astype(int)

    # 🔁 7. Group Orders by Week and Dish
    weekly_orders = orders_df.groupby(['Year', 'Week', 'DishName']).agg({
        'Quantity': 'sum',
        'Month': 'max',
        'IsWeekend': 'max',
        'IsHoliday': 'max'
    }).reset_index()

    weekly_orders['TrendIndex'] = weekly_orders.groupby('DishName').cumcount()

    # 🔮 8. Forecast Dish Quantities
    unique_dishes = weekly_orders['DishName'].unique()
    next_week_number = weekly_orders['Week'].max() + 1
    next_year = weekly_orders['Year'].max()
    next_month_number = 10  # Adjust if needed

    forecast_results = []

    for dish_name in unique_dishes:
        dish_data = weekly_orders[weekly_orders['DishName'] == dish_name]
        if len(dish_data) < 5:
            continue

        X = dish_data[['Week', 'Month', 'IsWeekend', 'IsHoliday', 'TrendIndex']].copy()
        y = dish_data['Quantity']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        if len(y_test) >= 2:
            mae = mean_absolute_error(y_test, model.predict(X_test))
            r2 = model.score(X_test, y_test)
        else:
            mae = None
            r2 = None

        next_week = pd.DataFrame({
            'Week': [next_week_number],
            'Month': [next_month_number],
            'IsWeekend': [1],
            'IsHoliday': [0],
            'TrendIndex': [len(dish_data)]
        })

        prediction = model.predict(next_week)[0]
        prediction = np.clip(prediction, 0, None)
        adjusted = min(round(prediction), 30)

        forecast_results.append({
            'DishName': dish_name,
            'ForecastQty': adjusted,
            'Week': next_week_number,
            'Year': next_year,
            'R2': round(r2, 3) if r2 is not None else None,
            'MAE': round(mae, 2) if mae is not None else None
        })

    return forecast_results