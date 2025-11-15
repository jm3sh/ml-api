#!/usr/bin/env python3
"""
Customer Behavior Analysis using Linear Regression
Analyzes customer spending patterns and predicts future behavior
"""

import sys, json, numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from collections import defaultdict

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
            clv = {'predicted_30day_value': round(pred,2), 'model_coefficients': {'orders_impact': round(model.coef_[0],2), 'days_impact': round(model.coef_[1],2)}}
        else:
            clv = {'predicted_30day_value': round(avg_order_value * 1.2, 2), 'model_coefficients': None}

        peak_hours = sorted([{'hour': h, 'order_count': c, 'hour_label': f'{h:02d}:00 - {h:02d}:59'} for h,c in hourly_orders.items()], key=lambda x: x['order_count'], reverse=True)[:5]
        peak_days = sorted([{'day': d, 'order_count': c} for d,c in daily_orders.items()], key=lambda x: x['order_count'], reverse=True)

        recs = []
        if repeat_rate < 30: recs.append("Low repeat rate detected. Consider implementing a loyalty program.")
        if peak_hours: recs.append(f"Peak ordering time is {peak_hours[0]['hour_label']}. Consider running promotions during this period.")
        if avg_order_value < 300: recs.append("Average order value is low. Consider upselling or bundle offers.")

        return {'success': True, 'insights': {'total_customers': total_customers, 'repeat_customers': repeat_customers, 'repeat_customer_rate': repeat_rate, 'avg_order_value': avg_order_value, 'peak_hours': peak_hours, 'peak_days': peak_days, 'customer_lifetime_value': clv}, 'recommendations': recs, 'algorithm': 'Linear Regression'}
    except Exception as e:
        return {'success': False, 'error': f'Error in customer insights: {str(e)}'}

def main():
    if len(sys.argv) < 2:
        print(json.dumps({'success': False, 'error': 'No input file provided'}))
        sys.exit(1)
    with open(sys.argv[1], 'r') as f:
        data = json.load(f)
    print(json.dumps(run(data)))

if __name__ == '__main__':
    main()
