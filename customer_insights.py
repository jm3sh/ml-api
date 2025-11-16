from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from collections import defaultdict
import os

app = Flask(__name__)

def load_csv_fallback():
    try:
        path = os.path.join(os.path.dirname(__file__), 'order_data.csv')
        df = pd.read_csv(path)
        df = df.rename(columns=str.strip)

        # Simulate customer_id and order_time
        df['customer_id'] = 1000 + df.index  # dummy IDs
        df['order_time'] = pd.to_datetime(df['OrderDate']).dt.strftime('%Y-%m-%d %H:%M:%S')
        df['order_amount'] = df['Quantity'] * 150  # assume ₱150 per dish

        return df[['customer_id', 'order_time', 'order_amount']].to_dict(orient='records')
    except Exception as e:
        return []

def run(data):
    try:
        if not data:
            return {'success': False, 'error': 'No customer data available'}

        customer_orders = defaultdict(list)
        all_amounts, hourly_orders, daily_orders = [], defaultdict(int), defaultdict(int)

        for order in data:
            dt = datetime.strptime(order['order_time'], '%Y-%m-%d %H:%M:%S')
            customer_orders[order['customer_id']].append({'time': dt, 'amount': order['order_amount']})
            all_amounts.append(order['order_amount'])
            hourly_orders[dt.hour] += 1
            daily_orders[dt.strftime('%A')] += 1

        total_customers = len(customer_orders)
        repeat_customers = sum(1 for o in customer_orders.values() if len(o) > 1)
        repeat_rate = round((repeat_customers / total_customers * 100) if total_customers else 0, 1)
        avg_order_value = round(np.mean(all_amounts), 2) if all_amounts else 0

        metrics = []
        for cid, orders in customer_orders.items():
            num = len(orders)
            spent = sum(o['amount'] for o in orders)
            avg = spent / num
            days = (max(o['time'] for o in orders) - min(o['time'] for o in orders)).days + 1 if num > 1 else 0
            metrics.append({'customer_id': cid, 'num_orders': num, 'total_spent': spent, 'avg_spend': avg, 'days_active': days})

        multi = [c for c in metrics if c['num_orders'] > 1]
        if len(multi) >= 3:
            X = np.array([[c['num_orders'], c['days_active']] for c in multi])
            y = np.array([c['total_spent'] for c in multi])
            model = LinearRegression().fit(X, y)
            pred = model.predict([[np.mean(X[:,0])+1, np.mean(X[:,1])+30]])[0]
            clv = {
                'predicted_30day_value': round(pred,2),
                'model_coefficients': {
                    'orders_impact': round(model.coef_[0],2),
                    'days_impact': round(model.coef_[1],2)
                }
            }
        else:
            clv = {
                'predicted_30day_value': round(avg_order_value * 1.2, 2),
                'model_coefficients': None
            }

        peak_hours = sorted([
            {'hour': h, 'order_count': c, 'hour_label': f'{h:02d}:00 - {h:02d}:59'}
            for h,c in hourly_orders.items()
        ], key=lambda x: x['order_count'], reverse=True)[:5]

        peak_days = sorted([
            {'day': d, 'order_count': c}
            for d,c in daily_orders.items()
        ], key=lambda x: x['order_count'], reverse=True)

        recs = []
        if repeat_rate < 30:
            recs.append("Low repeat rate detected. Consider implementing a loyalty program.")
        if peak_hours:
            recs.append(f"Peak ordering time is {peak_hours[0]['hour_label']}. Consider running promotions during this period.")
        if avg_order_value < 300:
            recs.append("Average order value is low. Consider upselling or bundle offers.")

        return {
            'success': True,
            'insights': {
                'total_customers': total_customers,
                'repeat_customers': repeat_customers,
                'repeat_customer_rate': repeat_rate,
                'avg_order_value': avg_order_value,
                'peak_hours': peak_hours,
                'peak_days': peak_days,
                'customer_lifetime_value': clv
            },
            'recommendations': recs,
            'algorithm': 'Linear Regression'
        }

    except Exception as e:
        return {'success': False, 'error': f'Error in customer insights: {str(e)}'}

@app.route('/customer_insights', methods=['POST'])
def customer_insights():
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
