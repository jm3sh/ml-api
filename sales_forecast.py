#!/usr/bin/env python3
"""
Sales Forecasting using Linear Regression
Predicts future orders and revenue based on historical trends
"""

import sys
import json
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

def main():
    if len(sys.argv) < 2:
        print(json.dumps({'success': False, 'error': 'No input file provided'}))
        sys.exit(1)
    
    try:
        # Read input data
        with open(sys.argv[1], 'r') as f:
            data = json.load(f)
        
        if not data or len(data) < 7:
            print(json.dumps({
                'success': False,
                'error': 'Insufficient data. Need at least 7 days of history.'
            }))
            sys.exit(0)
        
        # Prepare data
        dates = [datetime.strptime(d['date'], '%Y-%m-%d') for d in data]
        orders = np.array([d['orders'] for d in data])
        revenue = np.array([d['revenue'] for d in data])
        
        # Create day indices (0, 1, 2, ... for each day)
        day_indices = np.array([(d - dates[0]).days for d in dates]).reshape(-1, 1)
        
        # Train Linear Regression models
        order_model = LinearRegression()
        revenue_model = LinearRegression()
        
        order_model.fit(day_indices, orders)
        revenue_model.fit(day_indices, revenue)
        
        # Calculate model performance metrics
        order_predictions_train = order_model.predict(day_indices)
        revenue_predictions_train = revenue_model.predict(day_indices)
        
        order_r2 = r2_score(orders, order_predictions_train)
        revenue_r2 = r2_score(revenue, revenue_predictions_train)
        
        order_mae = mean_absolute_error(orders, order_predictions_train)
        revenue_mae = mean_absolute_error(revenue, revenue_predictions_train)
        
        # Generate 7-day forecast
        last_date = dates[-1]
        predictions = []
        
        for i in range(1, 8):
            forecast_date = last_date + timedelta(days=i)
            forecast_day_index = np.array([[(forecast_date - dates[0]).days]])
            
            pred_orders = max(0, int(round(order_model.predict(forecast_day_index)[0])))
            pred_revenue = max(0, revenue_model.predict(forecast_day_index)[0])
            
            predictions.append({
                'date': forecast_date.strftime('%Y-%m-%d'),
                'day': forecast_date.strftime('%A'),
                'predicted_orders': pred_orders,
                'predicted_revenue': round(pred_revenue, 2)
            })
        
        # Calculate trends
        total_pred_orders = sum(p['predicted_orders'] for p in predictions)
        total_pred_revenue = sum(p['predicted_revenue'] for p in predictions)
        
        # Calculate average of last 7 days for comparison
        recent_orders = orders[-7:].mean() * 7
        recent_revenue = revenue[-7:].mean() * 7
        
        order_trend = round(((total_pred_orders - recent_orders) / recent_orders * 100) if recent_orders > 0 else 0, 1)
        revenue_trend = round(((total_pred_revenue - recent_revenue) / recent_revenue * 100) if recent_revenue > 0 else 0, 1)
        
        # Convert R² to confidence percentage (0-100 scale)
        order_confidence = round(max(0, min(100, order_r2 * 100)), 1)
        revenue_confidence = round(max(0, min(100, revenue_r2 * 100)), 1)
        
        result = {
            'success': True,
            'predictions': predictions,
            'summary': {
                'total_predicted_orders': total_pred_orders,
                'total_predicted_revenue': round(total_pred_revenue, 2),
                'order_trend': order_trend,
                'revenue_trend': revenue_trend,
                'confidence_orders': order_confidence,
                'confidence_revenue': revenue_confidence
            },
            'model_info': {
                'algorithm': 'Linear Regression',
                'training_days': len(data),
                'order_slope': round(float(order_model.coef_[0]), 4),
                'revenue_slope': round(float(revenue_model.coef_[0]), 4),
                'order_intercept': round(float(order_model.intercept_), 2),
                'revenue_intercept': round(float(revenue_model.intercept_), 2),
                'order_mae': round(order_mae, 2),
                'revenue_mae': round(revenue_mae, 2)
            }
        }
        
        print(json.dumps(result))
        
    except Exception as e:
        print(json.dumps({
            'success': False,
            'error': f'Error in sales forecast: {str(e)}'
        }))
        sys.exit(1)

if __name__ == '__main__':
    main()