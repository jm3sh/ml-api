#!/usr/bin/env python3
"""
Customer Behavior Analysis using Linear Regression
Analyzes customer spending patterns and predicts future behavior
"""

import sys
import json
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from collections import defaultdict

def main():
    if len(sys.argv) < 2:
        print(json.dumps({'success': False, 'error': 'No input file provided'}))
        sys.exit(1)
    
    try:
        # Read input data
        with open(sys.argv[1], 'r') as f:
            data = json.load(f)
        
        if not data:
            print(json.dumps({
                'success': False,
                'error': 'No customer data available'
            }))
            sys.exit(0)
        
        # Parse data
        customer_orders = defaultdict(list)
        all_amounts = []
        hourly_orders = defaultdict(int)
        daily_orders = defaultdict(int)
        
        for order in data:
            customer_orders[order['customer_id']].append({
                'time': datetime.strptime(order['order_time'], '%Y-%m-%d %H:%M:%S'),
                'amount': order['order_amount']
            })
            all_amounts.append(order['order_amount'])
            
            order_time = datetime.strptime(order['order_time'], '%Y-%m-%d %H:%M:%S')
            hour = order_time.hour
            day = order_time.strftime('%A')
            
            hourly_orders[hour] += 1
            daily_orders[day] += 1
        
        # Calculate basic metrics
        total_customers = len(customer_orders)
        repeat_customers = sum(1 for orders in customer_orders.values() if len(orders) > 1)
        repeat_rate = round((repeat_customers / total_customers * 100) if total_customers > 0 else 0, 1)
        avg_order_value = round(np.mean(all_amounts), 2) if all_amounts else 0
        
        # Use Linear Regression to predict customer lifetime value
        # Based on number of orders and average spending
        customer_metrics = []
        for cid, orders in customer_orders.items():
            num_orders = len(orders)
            total_spent = sum(o['amount'] for o in orders)
            avg_spend = total_spent / num_orders
            
            # Time span of customer activity (in days)
            if num_orders > 1:
                first_order = min(o['time'] for o in orders)
                last_order = max(o['time'] for o in orders)
                days_active = (last_order - first_order).days + 1
            else:
                days_active = 0
            
            customer_metrics.append({
                'customer_id': cid,
                'num_orders': num_orders,
                'total_spent': total_spent,
                'avg_spend': avg_spend,
                'days_active': days_active
            })
        
        # Predict future spending using linear regression
        # (for customers with multiple orders)
        multi_order_customers = [c for c in customer_metrics if c['num_orders'] > 1]
        
        if len(multi_order_customers) >= 3:
            X = np.array([[c['num_orders'], c['days_active']] for c in multi_order_customers])
            y = np.array([c['total_spent'] for c in multi_order_customers])
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Predict for average customer (used for insights)
            avg_orders = np.mean([c['num_orders'] for c in multi_order_customers])
            avg_days = np.mean([c['days_active'] for c in multi_order_customers])
            predicted_ltv = model.predict([[avg_orders + 1, avg_days + 30]])[0]
            
            clv_info = {
                'predicted_30day_value': round(predicted_ltv, 2),
                'model_coefficients': {
                    'orders_impact': round(float(model.coef_[0]), 2),
                    'days_impact': round(float(model.coef_[1]), 2)
                }
            }
        else:
            clv_info = {
                'predicted_30day_value': round(avg_order_value * 1.2, 2),
                'model_coefficients': None
            }
        
        # Peak hours analysis
        peak_hours = sorted(
            [{'hour': h, 'order_count': c, 'hour_label': f'{h:02d}:00 - {h:02d}:59'} 
             for h, c in hourly_orders.items()],
            key=lambda x: x['order_count'],
            reverse=True
        )[:5]
        
        # Peak days analysis
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        peak_days = sorted(
            [{'day': d, 'order_count': c} for d, c in daily_orders.items()],
            key=lambda x: x['order_count'],
            reverse=True
        )[:7]
        
        # Generate recommendations
        recommendations = []
        
        if repeat_rate < 30:
            recommendations.append("Low repeat rate detected. Consider implementing a loyalty program.")
        if peak_hours:
            top_hour = peak_hours[0]['hour_label']
            recommendations.append(f"Peak ordering time is {top_hour}. Consider running promotions during this period.")
        if avg_order_value < 300:
            recommendations.append("Average order value is low. Consider upselling or bundle offers.")
        
        result = {
            'success': True,
            'insights': {
                'total_customers': total_customers,
                'repeat_customers': repeat_customers,
                'repeat_customer_rate': repeat_rate,
                'avg_order_value': avg_order_value,
                'peak_hours': peak_hours,
                'peak_days': peak_days,
                'customer_lifetime_value': clv_info
            },
            'recommendations': recommendations,
            'algorithm': 'Linear Regression'
        }
        
        print(json.dumps(result))
        
    except Exception as e:
        print(json.dumps({
            'success': False,
            'error': f'Error in customer insights: {str(e)}'
        }))
        sys.exit(1)

if __name__ == '__main__':
    main()