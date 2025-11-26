import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from datetime import datetime
import os

def generate_forecast():
    try:
        # Load CSVs safely
        orders_path = os.path.join(os.getcwd(), 'final23_order_data.csv')
        ingredients_path = os.path.join(os.getcwd(), 'dishingredients.csv')

        if not os.path.exists(orders_path) or not os.path.exists(ingredients_path):
            return {
                "success": False,
                "error": "CSV files not found. Ensure final23_order_data.csv and dishingredients.csv are present."
            }

        orders_df = pd.read_csv(orders_path)
        ingredients_df = pd.read_csv(ingredients_path)

        orders_df.columns = [str(col).strip() for col in orders_df.columns]
        ingredients_df.columns = [str(col).strip() for col in ingredients_df.columns]

        orders_df['OrderDate'] = pd.to_datetime(orders_df['OrderDate'])
        orders_df['Year'] = orders_df['OrderDate'].dt.year
        current_year = datetime.now().year
        orders_df = orders_df[orders_df['Year'] == current_year]
        orders_df['Week'] = orders_df['OrderDate'].dt.isocalendar().week
        orders_df['Month'] = orders_df['OrderDate'].dt.month
        orders_df['DayOfWeek'] = orders_df['OrderDate'].dt.dayofweek
        orders_df['IsWeekend'] = orders_df['DayOfWeek'].isin([5, 6]).astype(int)

        holiday_dates = pd.to_datetime([
            '2025-01-01','2025-01-29','2025-04-09','2025-04-17','2025-04-18','2025-04-19',
            '2025-05-01','2025-06-12','2025-08-21','2025-08-25','2025-10-31','2025-11-01',
            '2025-11-30','2025-12-08','2025-12-24','2025-12-25','2025-12-30','2025-12-31'
        ])
        orders_df['IsHoliday'] = orders_df['OrderDate'].isin(holiday_dates).astype(int)

        weekly_orders = orders_df.groupby(['Year','Week','DishName']).agg({
            'Quantity':'sum','Month':'max','IsWeekend':'max','IsHoliday':'max'
        }).reset_index()
        weekly_orders['TrendIndex'] = weekly_orders.groupby('DishName').cumcount()

        unique_dishes = weekly_orders['DishName'].unique()
        next_week_number = datetime.now().isocalendar().week
        next_month_number = datetime.now().month

        forecast_results = []
        for dish_name in unique_dishes:
            dish_data = weekly_orders[weekly_orders['DishName'] == dish_name]
            if len(dish_data) < 5:
                continue

            X = dish_data[['Week','Month','IsWeekend','IsHoliday','TrendIndex']]
            y = dish_data['Quantity']

            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
            model = LinearRegression()
            model.fit(X_train,y_train)

            mae = mean_absolute_error(y_test, model.predict(X_test)) if len(y_test) >= 2 else None
            r2 = model.score(X_test, y_test) if len(y_test) >= 2 else None

            next_week = pd.DataFrame({
                'Week':[next_week_number],
                'Month':[next_month_number],
                'IsWeekend':[1],
                'IsHoliday':[0],
                'TrendIndex':[len(dish_data)]
            })

            prediction = model.predict(next_week)[0]
            adjusted = min(round(np.clip(prediction,0,None)),30)

            forecast_results.append({
                'DishName': dish_name,
                'ForecastQty': adjusted,
                'Week': next_week_number,
                'Year': current_year,
                'R2': round(r2,3) if r2 is not None else None,
                'MAE': round(mae,2) if mae is not None else None
            })

        forecast_df = pd.DataFrame(forecast_results)
        merged_df = pd.merge(forecast_df, ingredients_df, on='DishName', how='left')

        def assign_unit(ingredient):
            ingredient_lower = str(ingredient).lower()
            if any(word in ingredient_lower for word in ['beef','pork','chicken','bangus','fish','seafood','meat','tapa','leg']):
                return 'kg'
            elif any(word in ingredient_lower for word in ['broccoli','vegetable','mushroom','onion','leaves']):
                return 'kg'
            elif any(word in ingredient_lower for word in ['milk','oil','sauce','vinegar','water']):
                return 'L'
            elif any(word in ingredient_lower for word in ['flour','rice','sugar']):
                return 'kg'
            elif any(word in ingredient_lower for word in ['cheese','butter']):
                return 'kg'
            elif 'egg' in ingredient_lower:
                return 'pcs'
            elif any(word in ingredient_lower for word in ['wrapper','jelly']):
                return 'pack'
            else:
                return 'unit'

        merged_df['Unit'] = merged_df['Ingredient'].apply(assign_unit)
        merged_df['TotalIngredientQty'] = merged_df['ForecastQty'] * merged_df['QuantityRequired']
        ingredient_totals = merged_df.groupby(['Ingredient','Unit'])['TotalIngredientQty'].sum().reset_index()
        ingredient_totals['TotalIngredientQty'] = ingredient_totals['TotalIngredientQty'].round(2)

        return {
            "dishes": forecast_results,
            "ingredients": ingredient_totals.to_dict(orient="records")
        }

    except Exception as e:
        return {"success": False, "error": f"Error generating forecast: {str(e)}"}

