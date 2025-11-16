#!/usr/bin/env python3
"""
Anomaly Detection using Statistical Methods
Linear Regression is used to establish baseline trends,
then statistical deviation (Z-score) identifies anomalies
"""

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
import os

app = Flask(__name__)

def load_csv_fallback():
    try:
        path = os.path.join(os.path.dirname(__file__), 'order_data.csv')
        df = pd.read_csv(path)
        df = df.rename(columns=str.strip)  # Clean column names
        df['date'] = pd.to_datetime(df['OrderDate']).dt.strftime('%Y-%m-%d')
        df['orders'] = df['Quantity']
        df['revenue'] = df['Quantity'] * 150  # Example: assume ₱150 per dish
        return df[['date', 'orders', 'revenue']].to_dict(orient='records')
    except Exception as e:
        return []

def run(data):
    try:
        if len(data) < 14:
            return {
                'success': False,
                'error': 'Insufficient data for anomaly detection (minimum 14 days required)'
            }

        dates = [datetime.strptime(d['date'], '%Y-%m-%d') for d in data]
        orders = np.array([d['orders'] for d in data])
        revenue = np.array([d['revenue'] for d in data])
        day_indices = np.array([(d - dates[0]).days for d in dates]).reshape(-1, 1)

        order_model = LinearRegression()
        revenue_model = LinearRegression()
        order_model.fit(day_indices, orders)
        revenue_model.fit(day_indices, revenue)

        expected_orders = order_model.predict(day_indices)
        expected_revenue = revenue_model.predict(day_indices)

        order_residuals = orders - expected_orders
        revenue_residuals = revenue - expected_revenue

        order_std = np.std(order_residuals)
        revenue_std = np.std(revenue_residuals)

        anomalies = []
        high_severity_count = 0

        for i, (date, actual_orders, actual_revenue, exp_orders, exp_revenue) in enumerate(
            zip(dates, orders, revenue, expected_orders, expected_revenue)
        ):
            order_zscore = abs(order_residuals[i] / order_std) if order_std > 0 else 0
            revenue_zscore = abs(revenue_residuals[i] / revenue_std) if revenue_std > 0 else 0

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

        severity_order = {'high': 0, 'medium': 1}
        anomalies.sort(key=lambda x: (severity_order[x['severity']], x['date']), reverse=True)

        return {
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

    except Exception as e:
        return {
            'success': False,
            'error': f'Error in anomaly detection: {str(e)}'
        }

@app.route('/anomaly_detection', methods=['POST'])
def anomaly_detection():
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
