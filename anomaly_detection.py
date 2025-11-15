#!/usr/bin/env python3
"""
Anomaly Detection using Statistical Methods
Linear Regression is used to establish baseline trends,
then statistical deviation (Z-score) identifies anomalies
"""

import sys
import json
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression

def main():
    if len(sys.argv) < 2:
        print(json.dumps({'success': False, 'error': 'No input file provided'}))
        sys.exit(1)
    
    try:
        # Read input data
        with open(sys.argv[1], 'r') as f:
            data = json.load(f)
        
        if len(data) < 14:
            print(json.dumps({
                'success': False,
                'error': 'Insufficient data for anomaly detection (minimum 14 days required)'
            }))
            sys.exit(0)
        
        # Parse data
        dates = [datetime.strptime(d['date'], '%Y-%m-%d') for d in data]
        orders = np.array([d['orders'] for d in data])
        revenue = np.array([d['revenue'] for d in data])
        
        # Create day indices
        day_indices = np.array([(d - dates[0]).days for d in dates]).reshape(-1, 1)
        
        # Fit linear regression to establish baseline trend
        order_model = LinearRegression()
        revenue_model = LinearRegression()
        
        order_model.fit(day_indices, orders)
        revenue_model.fit(day_indices, revenue)
        
        # Predict expected values
        expected_orders = order_model.predict(day_indices)
        expected_revenue = revenue_model.predict(day_indices)
        
        # Calculate residuals (actual - predicted)
        order_residuals = orders - expected_orders
        revenue_residuals = revenue - expected_revenue
        
        # Calculate standard deviation of residuals
        order_std = np.std(order_residuals)
        revenue_std = np.std(revenue_residuals)
        
        # Detect anomalies using Z-score (threshold: 2 standard deviations)
        anomalies = []
        high_severity_count = 0
        
        for i, (date, actual_orders, actual_revenue, exp_orders, exp_revenue) in enumerate(
            zip(dates, orders, revenue, expected_orders, expected_revenue)
        ):
            order_zscore = abs(order_residuals[i] / order_std) if order_std > 0 else 0
            revenue_zscore = abs(revenue_residuals[i] / revenue_std) if revenue_std > 0 else 0
            
            # Check for order anomalies
            if order_zscore > 2.0:
                severity = 'high' if order_zscore > 3.0 else 'medium'
                if severity == 'high':
                    high_severity_count += 1
                
                direction = 'spike' if order_residuals[i] > 0 else 'drop'
                
                anomalies.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'type': 'orders',
                    'description': f"Unusual {direction} in order volume",
                    'value': int(actual_orders),
                    'expected': float(exp_orders),
                    'deviation': round(order_zscore, 2),
                    'severity': severity
                })
            
            # Check for revenue anomalies
            if revenue_zscore > 2.0:
                severity = 'high' if revenue_zscore > 3.0 else 'medium'
                if severity == 'high':
                    high_severity_count += 1
                
                direction = 'spike' if revenue_residuals[i] > 0 else 'drop'
                
                anomalies.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'type': 'revenue',
                    'description': f"Unusual {direction} in revenue",
                    'value': float(actual_revenue),
                    'expected': float(exp_revenue),
                    'deviation': round(revenue_zscore, 2),
                    'severity': severity
                })
        
        # Sort by severity and date
        severity_order = {'high': 0, 'medium': 1}
        anomalies.sort(key=lambda x: (severity_order[x['severity']], x['date']), reverse=True)
        
        result = {
            'success': True,
            'anomalies': anomalies,
            'summary': {
                'total_anomalies': len(anomalies),
                'high_severity': high_severity_count,
                'medium_severity': len(anomalies) - high_severity_count,
                'days_analyzed': len(data)
            },
            'model_info': {
                'algorithm': 'Linear Regression + Z-score Analysis',
                'order_baseline_slope': round(float(order_model.coef_[0]), 4),
                'revenue_baseline_slope': round(float(revenue_model.coef_[0]), 4),
                'threshold': '2 standard deviations'
            }
        }
        
        print(json.dumps(result))
        
    except Exception as e:
        print(json.dumps({
            'success': False,
            'error': f'Error in anomaly detection: {str(e)}'
        }))
        sys.exit(1)

if __name__ == '__main__':
    main()