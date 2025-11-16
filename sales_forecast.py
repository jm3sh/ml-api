from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import os

app = Flask(__name__)

def load_csv_fallback():
    try:
        path = os.path.join(os.path.dirname(__file__), 'order_data.csv')
        df = pd.read_csv(path)
        df = df.rename(columns=str.strip)

        df['date'] = pd.to_datetime(df['OrderDate']).dt.strftime('%Y-%m-%d')
        df['orders'] = df['Quantity']
        df['revenue'] = df['Quantity'] * 150  # assume ₱150 per dish

        return df[['date', 'orders', 'revenue']].to_dict(orient='records')
    except Exception as e:
        return []

def run(data):
    try:
        if not data or len(data) < 7:
            return {'success': False, 'error': 'Insufficient data. Need at least 7 days of history.'}

        dates = [datetime.strptime(d['date'], '%Y-%m-%d') for d in data]
        orders = np.array([d['orders'] for d in data])
        revenue = np.array([d['revenue'] for d in data])
        X = np.array([(d - dates[0]).days for d in dates]).reshape(-1,1)

        om, rm = LinearRegression(), LinearRegression()
        om.fit(X, orders)
        rm.fit(X, revenue)

        pred_orders = om.predict(X)
        pred_revenue = rm.predict(X)

        order_r2 = r2_score(orders, pred_orders)
        revenue_r2 = r2_score(revenue, pred_revenue)
        order_mae = mean_absolute_error(orders, pred_orders)
        revenue_mae = mean_absolute_error(revenue, pred_revenue)

        forecast = []
        for i in range(1,8):
            fd = dates[-1] + timedelta(days=i)
            xi = np.array([[(fd - dates[0]).days]])
            forecast.append({
                'date': fd.strftime('%Y-%m-%d'),
                'day': fd.strftime('%A'),
                'predicted_orders': max(0,int(round(om.predict(xi)[0]))),
                'predicted_revenue': round(rm.predict(xi)[0],2)
            })

        total_orders = sum(f['predicted_orders'] for f in forecast)
        total_revenue = sum(f['predicted_revenue'] for f in forecast)
        recent_orders = orders[-7:].mean() * 7
        recent_revenue = revenue[-7:].mean() * 7

        return {
            'success': True,
            'predictions': forecast,
            'summary': {
                'total_predicted_orders': total_orders,
                'total_predicted_revenue': round(total_revenue,2),
                'order_trend': round((total_orders - recent_orders)/recent_orders*100 if recent_orders else 0,1),
                'revenue_trend': round((total_revenue - recent_revenue)/recent_revenue*100 if recent_revenue else 0,1),
                'confidence_orders': round(order_r2*100,1),
                'confidence_revenue': round(revenue_r2*100,1)
            },
            'model_info': {
                'algorithm': 'Linear Regression',
                'training_days': len(data),
                'order_slope': round(float(om.coef_[0]),4),
                'revenue_slope': round(float(rm.coef_[0]),4),
                'order_intercept': round(float(om.intercept_),2),
                'revenue_intercept': round(float(rm.intercept_),2),
                'order_mae': round(order_mae,2),
                'revenue_mae': round(revenue_mae,2)
            }
        }
    except Exception as e:
        return {'success': False, 'error': f'Error in sales forecast: {str(e)}'}

@app.route('/sales_forecast', methods=['POST'])
def sales_forecast():
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
