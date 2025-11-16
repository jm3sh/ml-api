from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from collections import defaultdict
import os

app = Flask(__name__)

def load_csv_fallback():
    try:
        path = os.path.join(os.path.dirname(__file__), 'order_data.csv')
        df = pd.read_csv(path)
        df = df.rename(columns=str.strip)

        # Simulate product_id and stock
        df['product_id'] = 1
        df['product_name'] = df['DishName']
        df['current_stock'] = 10  # dummy stock
        df['date'] = pd.to_datetime(df['OrderDate']).dt.strftime('%Y-%m-%d')
        df['quantity_sold'] = df['Quantity']

        return df[['product_id', 'product_name', 'current_stock', 'date', 'quantity_sold']].to_dict(orient='records')
    except Exception as e:
        return []

def run(data):
    try:
        if not data:
            return {'success': False, 'error': 'No product sales data available'}

        product_data = defaultdict(lambda: {'dates': [], 'quantities': [], 'name': '', 'stock': 0})

        for item in data:
            pid = item['product_id']
            product_data[pid]['name'] = item['product_name']
            product_data[pid]['stock'] = item['current_stock']
            product_data[pid]['dates'].append(datetime.strptime(item['date'], '%Y-%m-%d'))
            product_data[pid]['quantities'].append(item['quantity_sold'])

        results = []
        high_priority = 0
        total_restock = 0

        for pid, pdata in product_data.items():
            if len(pdata['dates']) < 3:
                continue

            sorted_indices = np.argsort(pdata['dates'])
            dates = [pdata['dates'][i] for i in sorted_indices]
            quantities = np.array([pdata['quantities'][i] for i in sorted_indices])
            day_indices = np.array([(d - dates[0]).days for d in dates]).reshape(-1, 1)

            model = LinearRegression()
            model.fit(day_indices, quantities)

            last_date = dates[-1]
            weekly_predictions = []
            for i in range(1, 8):
                forecast_date = last_date + timedelta(days=i)
                forecast_day_index = np.array([[(forecast_date - dates[0]).days]])
                pred = max(0, model.predict(forecast_day_index)[0])
                weekly_predictions.append(pred)

            weekly_demand = int(round(sum(weekly_predictions)))
            daily_avg = round(np.mean(quantities), 1)
            slope = float(model.coef_[0])
            avg_quantity = np.mean(quantities)
            trend_pct = round((slope / avg_quantity * 100) if avg_quantity > 0 else 0, 1)

            restock_needed = max(0, weekly_demand - int(pdata['stock']))
            stock_to_demand_ratio = pdata['stock'] / weekly_demand if weekly_demand > 0 else 999

            if stock_to_demand_ratio < 0.5:
                priority = 'high'
                high_priority += 1
            elif stock_to_demand_ratio < 1.0:
                priority = 'medium'
            else:
                priority = 'low'

            total_restock += restock_needed

            results.append({
                'product_id': pid,
                'product_name': pdata['name'],
                'current_stock': int(pdata['stock']),
                'predicted_weekly_demand': weekly_demand,
                'daily_average': daily_avg,
                'trend': trend_pct,
                'recommended_restock': restock_needed,
                'priority': priority,
                'days_of_stock': round(stock_to_demand_ratio * 7, 1) if weekly_demand > 0 else 999,
                'model_info': {
                    'slope': round(slope, 4),
                    'intercept': round(float(model.intercept_), 2),
                    'data_points': len(dates)
                }
            })

        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        results.sort(key=lambda x: (priority_order[x['priority']], -x['recommended_restock']))

        return {
            'success': True,
            'products': results,
            'summary': {
                'total_products': len(results),
                'high_priority_items': high_priority,
                'total_restock_needed': total_restock
            },
            'algorithm': 'Linear Regression'
        }

    except Exception as e:
        return {'success': False, 'error': f'Error in product demand prediction: {str(e)}'}

@app.route('/product_demand', methods=['POST'])
def product_demand():
    try:
        data = request.get_json()
        if not data:
            data = load_csv_fallback()
        result = run(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run()
